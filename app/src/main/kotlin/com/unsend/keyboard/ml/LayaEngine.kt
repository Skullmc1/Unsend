package com.unsend.keyboard.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.TimeUnit

/**
 * Manages the on-device Laya ONNX Runtime session and single-forward-pass tone/risk inference.
 */
class LayaEngine(context: Context) {

    private val appContext = context.applicationContext
    private val modelsDir = File(appContext.filesDir, "models").apply { mkdirs() }
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val sessionMutex = Mutex()
    private val lifecycleScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var tokenizer: Tokenizer = Tokenizer()
    private var unloadJob: Job? = null

    fun getModelsDir(): File = modelsDir

    fun findOnnxModelFile(): File? {
        val preferred = File(modelsDir, DEFAULT_MODEL_FILENAME)
        if (preferred.exists() && preferred.length() > 0L) return preferred
        return modelsDir.listFiles()?.firstOrNull {
            it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 0L
        }
    }

    fun findVocabFile(): File? {
        val tokJson = File(modelsDir, DEFAULT_TOKENIZER_FILENAME)
        if (tokJson.exists() && tokJson.length() > 0L) return tokJson
        val vocab = File(modelsDir, DEFAULT_VOCAB_FILENAME)
        return if (vocab.exists() && vocab.length() > 0L) vocab else null
    }

    fun isPreviewModeEnabled(): Boolean = prefs.getBoolean(KEY_PREVIEW_MODE, true)

    fun setPreviewModeEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_PREVIEW_MODE, enabled).apply()
    }

    fun getStatus(): ModelStatus = when {
        findOnnxModelFile() != null -> ModelStatus.READY_ONNX
        isPreviewModeEnabled() -> ModelStatus.PREVIEW_MODE
        else -> ModelStatus.NOT_CONFIGURED
    }

    suspend fun downloadModel(
        modelUrl: String = DEFAULT_MODEL_URL,
        tokenizerUrl: String = DEFAULT_TOKENIZER_URL,
        onProgress: (Float) -> Unit,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val modelDest = File(modelsDir, DEFAULT_MODEL_FILENAME)
            val tokenizerDest = File(modelsDir, DEFAULT_TOKENIZER_FILENAME)

            downloadFileWithProgress(tokenizerUrl, tokenizerDest) { p ->
                onProgress(p * 0.04f)
            }
            downloadFileWithProgress(modelUrl, modelDest) { p ->
                onProgress(0.04f + p * 0.96f)
            }
            onProgress(1f)
        }
    }

    private fun downloadFileWithProgress(
        urlString: String,
        destination: File,
        onProgress: (Float) -> Unit,
    ) {
        val tempFile = File(destination.parentFile, "${destination.name}.tmp")
        val conn = (URL(urlString).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
        }
        try {
            val code = conn.responseCode
            if (code !in 200..299) {
                error("HTTP $code from model host")
            }
            val totalBytes = conn.contentLengthLong.coerceAtLeast(1L)
            var downloaded = 0L
            conn.inputStream.use { input ->
                tempFile.outputStream().use { output ->
                    val buffer = ByteArray(16 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        downloaded += read
                        onProgress((downloaded.toFloat() / totalBytes).coerceIn(0f, 1f))
                    }
                }
            }
            if (!tempFile.renameTo(destination)) {
                tempFile.copyTo(destination, overwrite = true)
                tempFile.delete()
            }
        } catch (t: Throwable) {
            tempFile.delete()
            throw t
        } finally {
            conn.disconnect()
        }
    }

    suspend fun ensureLoaded() {
        unloadJob?.cancel()
        unloadJob = null

        val modelFile = findOnnxModelFile() ?: return
        sessionMutex.withLock {
            if (ortSession != null) return
            withContext(Dispatchers.IO) {
                runCatching {
                    val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                    val options = OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                        setIntraOpNumThreads(4)
                    }
                    ortSession = env.createSession(modelFile.absolutePath, options)
                    val vocabFile = findVocabFile()
                    tokenizer = when {
                        vocabFile == null -> Tokenizer()
                        vocabFile.name.endsWith(".json", ignoreCase = true) -> Tokenizer.fromTokenizerJson(vocabFile)
                        else -> Tokenizer.fromVocabFile(vocabFile)
                    }
                }.onFailure { err ->
                    Log.e(TAG, "Failed to load ONNX session", err)
                    ortSession = null
                }
            }
        }
    }

    fun scheduleUnload(gracePeriodMs: Long = UNLOAD_GRACE_MS) {
        unloadJob?.cancel()
        unloadJob = lifecycleScope.launch {
            delay(gracePeriodMs)
            unloadNow()
        }
    }

    suspend fun unloadNow() {
        sessionMutex.withLock {
            ortSession?.close()
            ortSession = null
        }
    }

    suspend fun analyze(
        draft: String,
        incomingReply: String?,
        preset: ContextPreset,
    ): LayaAnalysis = withContext(Dispatchers.Default) {
        val startMs = SystemClock.elapsedRealtime()
        val modelFile = findOnnxModelFile()

        if (modelFile != null) {
            ensureLoaded()
            val session = ortSession
            val env = ortEnv
            if (session != null && env != null) {
                val onnxResult = runCatching {
                    runOnnxInference(
                        env = env,
                        session = session,
                        draft = draft,
                        incomingReply = incomingReply,
                        preset = preset,
                        startMs = startMs,
                    )
                }.onFailure { err ->
                    Log.e(TAG, "ONNX inference failed, falling back to built-in analyzer", err)
                }.getOrNull()
                if (onnxResult != null) return@withContext onnxResult
            }
        }

        runPreviewHeuristic(
            draft = draft,
            incomingReply = incomingReply,
            preset = preset,
            startMs = startMs,
        )
    }

    private suspend fun runOnnxInference(
        env: OrtEnvironment,
        session: OrtSession,
        draft: String,
        incomingReply: String?,
        preset: ContextPreset,
        startMs: Long,
    ): LayaAnalysis {
        val expectedInputs = session.inputNames
        if ("marker_pos" in expectedInputs && "marker_mask" in expectedInputs && "qtype" in expectedInputs) {
            return runLayaBatchedInference(env, session, draft, incomingReply, preset, startMs)
        }

        val prompt = ToneSchema.buildPrompt(draft, incomingReply, preset)
        val tokenized = tokenizer.tokenize(prompt)

        val rawLogits: FloatArray = sessionMutex.withLock {
            val inputIdsTensor = OnnxTensor.createTensor(env, arrayOf(tokenized.inputIds))
            val maskTensor = OnnxTensor.createTensor(env, arrayOf(tokenized.attentionMask))

            inputIdsTensor.use { ids ->
                maskTensor.use { mask ->
                    val inputs = buildMap {
                        if ("input_ids" in expectedInputs) put("input_ids", ids)
                        if ("attention_mask" in expectedInputs) put("attention_mask", mask)
                        if (isEmpty() && expectedInputs.isNotEmpty()) {
                            put(expectedInputs.first(), ids)
                        }
                    }

                    session.run(inputs).use { results ->
                        extractFloatVector(results[0].value)
                    }
                }
            }
        }

        return decodeLogits(
            logits = rawLogits,
            hasReplyContext = !incomingReply.isNullOrBlank(),
            preset = preset,
            elapsedMs = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(1L),
        )
    }

    private data class LayaQuestion(
        val qtype: Long,
        val qtypeName: String,
        val instructions: String,
        val options: List<String>,
    )

    private suspend fun runLayaBatchedInference(
        env: OrtEnvironment,
        session: OrtSession,
        draft: String,
        incomingReply: String?,
        preset: ContextPreset,
        startMs: Long,
    ): LayaAnalysis {
        val stateText = buildString {
            append("Context: ").append(preset.label).append('\n')
            if (!incomingReply.isNullOrBlank()) {
                append("Incoming message: ").append(incomingReply.trim()).append('\n')
            }
            append("Draft message: ").append(draft.trim())
        }
        val stateIds = tokenizer.encode(stateText)

        val questions = buildList {
            // 0: Escalation score (5 levels -> 0..1)
            add(
                LayaQuestion(
                    qtype = 1L,
                    qtypeName = "score",
                    instructions = "Rate the escalation or regret risk of sending this draft message.",
                    options = listOf(
                        "level 0: calm, constructive, no regret risk",
                        "level 1: mild tension or abruptness",
                        "level 2: noticeable defensiveness or friction",
                        "level 3: confrontational or likely to escalate",
                        "level 4: severe hostility or high regret risk",
                    ),
                ),
            )
            // 1: Primary Tone (6 choices matching ToneLabel order)
            add(
                LayaQuestion(
                    qtype = 0L,
                    qtypeName = "choice",
                    instructions = "What is the primary tone of the draft message?",
                    options = listOf(
                        "Warm: friendly, empathetic, or appreciative",
                        "Assertive: clear, direct, and respectful",
                        "Defensive: self-justifying or reactive",
                        "Passive-Aggressive: subtle hostility or sarcasm",
                        "Blunt: curt, harsh, or lacking tact",
                        "Unclear: ambiguous or confusing",
                    ),
                ),
            )
            // 2: Clarity (noul: [false, true])
            add(
                LayaQuestion(
                    qtype = 2L,
                    qtypeName = "noul",
                    instructions = "The draft message is clear and easy to understand.",
                    options = listOf(
                        "false: no, the statement does not hold",
                        "true: yes, the statement holds",
                    ),
                ),
            )
            // 3: Answered question? (Reply Mode only)
            if (!incomingReply.isNullOrBlank()) {
                add(
                    LayaQuestion(
                        qtype = 2L,
                        qtypeName = "noul",
                        instructions = "The draft message directly answers or addresses the incoming message.",
                        options = listOf(
                            "false: no, the statement does not hold",
                            "true: yes, the statement holds",
                        ),
                    ),
                )
            }
        }

        val maxSeqLen = 256
        val maxMarkers = 6
        val batch = questions.size

        val qtypes = LongArray(batch)
        var actualMaxSeq = 1

        val encodedSequences = questions.mapIndexed { row, q ->
            qtypes[row] = q.qtype
            val headIds = tokenizer.encode("${q.qtypeName} question: ${q.instructions}")
            val seq = ArrayList<Int>(maxSeqLen)
            seq.add(tokenizer.clsId)
            seq.addAll(headIds.take(48))
            seq.add(tokenizer.sepId)

            val markers = mutableListOf<Int>()
            for (opt in q.options) {
                if (seq.size >= maxSeqLen - 16) break
                markers.add(seq.size)
                seq.add(tokenizer.maskId)
                seq.addAll(tokenizer.encode(" $opt").take(24))
            }
            seq.add(tokenizer.sepId)

            val room = (maxSeqLen - seq.size - 1).coerceAtLeast(0)
            seq.addAll(stateIds.take(room))
            seq.add(tokenizer.sepId)

            val len = seq.size.coerceAtMost(maxSeqLen)
            if (len > actualMaxSeq) actualMaxSeq = len
            seq.take(len) to markers
        }

        // Pack tightly to actualMaxSeq as 2D arrays for safe OnnxTensor creation
        val packedIds = Array(batch) { LongArray(actualMaxSeq) { tokenizer.padId.toLong() } }
        val packedMask = Array(batch) { LongArray(actualMaxSeq) { 0L } }
        val markerPosArr = Array(batch) { LongArray(maxMarkers) { 0L } }
        val markerMaskArr = Array(batch) { BooleanArray(maxMarkers) { false } }

        encodedSequences.forEachIndexed { row, (seq, markers) ->
            for (i in seq.indices) {
                packedIds[row][i] = seq[i].toLong()
                packedMask[row][i] = 1L
            }
            for (mIdx in markers.indices) {
                if (mIdx < maxMarkers) {
                    markerPosArr[row][mIdx] = markers[mIdx].toLong()
                    markerMaskArr[row][mIdx] = true
                }
            }
        }

        val logitsRows: Array<FloatArray> = sessionMutex.withLock {
            val idsTensor = OnnxTensor.createTensor(env, packedIds)
            val maskTensor = OnnxTensor.createTensor(env, packedMask)
            val mPosTensor = OnnxTensor.createTensor(env, markerPosArr)
            val mMaskTensor = OnnxTensor.createTensor(env, markerMaskArr)
            val qtypeTensor = OnnxTensor.createTensor(env, qtypes)

            idsTensor.use { t0 ->
                maskTensor.use { t1 ->
                    mPosTensor.use { t2 ->
                        mMaskTensor.use { t3 ->
                            qtypeTensor.use { t4 ->
                                val inputs = mapOf(
                                    "input_ids" to t0,
                                    "attention_mask" to t1,
                                    "marker_pos" to t2,
                                    "marker_mask" to t3,
                                    "qtype" to t4,
                                )
                                session.run(inputs).use { results ->
                                    @Suppress("UNCHECKED_CAST")
                                    results[0].value as Array<FloatArray>
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply Laya calibrated temperature scaling
        val escLogits = FloatArray(5) { i -> logitsRows[0][i] / TEMP_SCORE }
        val escProbs = ToneSchema.softmax(escLogits)
        val escalationRisk = (0 until 5).sumOf { i -> (i / 4.0) * escProbs[i] }.toFloat().coerceIn(0f, 1f)

        val toneLogits = FloatArray(6) { i -> logitsRows[1][i] / TEMP_CHOICE_6 }
        val toneProbs = ToneSchema.softmax(toneLogits)
        val toneMap = ToneLabel.entries.mapIndexed { idx, label -> label to toneProbs[idx] }.toMap()
        val primaryTone = toneMap.maxByOrNull { it.value }?.key ?: ToneLabel.WARM

        val clarityLogits = FloatArray(2) { i -> logitsRows[2][i] / TEMP_NOUL }
        val clarityProbs = ToneSchema.softmax(clarityLogits)
        val clarityScore = clarityProbs[1]

        val answeredScore = if (batch >= 4) {
            val ansLogits = FloatArray(2) { i -> logitsRows[3][i] / TEMP_NOUL }
            ToneSchema.softmax(ansLogits)[1]
        } else {
            null
        }

        val elapsed = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(1L)
        return LayaAnalysis(
            escalationRisk = escalationRisk,
            riskLevel = ToneSchema.classifyRisk(escalationRisk, preset),
            primaryTone = primaryTone,
            toneProbabilities = toneMap,
            clarityScore = clarityScore,
            answeredQuestionScore = answeredScore,
            inferenceTimeMs = elapsed,
            isPreviewHeuristic = false,
        )
    }

    private fun extractFloatVector(value: Any?): FloatArray = when (value) {
        is Array<*> -> {
            val first = value.firstOrNull()
            if (first is FloatArray) first else FloatArray(0)
        }
        is FloatArray -> value
        else -> FloatArray(0)
    }

    /**
     * Decodes a 9-element single-pass logit vector:
     * [0] escalation_logit, [1..6] tone_logits (6), [7] clarity_logit, [8] answered_question_logit.
     */
    private fun decodeLogits(
        logits: FloatArray,
        hasReplyContext: Boolean,
        preset: ContextPreset,
        elapsedMs: Long,
    ): LayaAnalysis {
        val escalationRisk = if (logits.isNotEmpty()) ToneSchema.sigmoid(logits[0]) else 0.2f
        val toneSlice = if (logits.size >= 7) logits.copyOfRange(1, 7) else FloatArray(6) { 0f }
        val toneProbsArray = ToneSchema.softmax(toneSlice)
        val toneMap = ToneLabel.entries.mapIndexed { idx, label ->
            label to toneProbsArray.getOrElse(idx) { 0f }
        }.toMap()
        val primaryTone = toneMap.maxByOrNull { it.value }?.key ?: ToneLabel.UNCLEAR
        val clarityScore = if (logits.size >= 8) ToneSchema.sigmoid(logits[7]) else 0.75f
        val answeredScore = if (hasReplyContext && logits.size >= 9) {
            ToneSchema.sigmoid(logits[8])
        } else {
            null
        }

        return LayaAnalysis(
            escalationRisk = escalationRisk,
            riskLevel = ToneSchema.classifyRisk(escalationRisk, preset),
            primaryTone = primaryTone,
            toneProbabilities = toneMap,
            clarityScore = clarityScore,
            answeredQuestionScore = answeredScore,
            inferenceTimeMs = elapsedMs,
            isPreviewHeuristic = false,
        )
    }

    /**
     * Lightweight on-device preview analyzer used when the user enables Preview Mode
     * before importing a quantized Laya .onnx model.
     */
    private fun runPreviewHeuristic(
        draft: String,
        incomingReply: String?,
        preset: ContextPreset,
        startMs: Long,
    ): LayaAnalysis {
        val trimmed = draft.trim()
        val lower = trimmed.lowercase()
        val words = trimmed.split(Regex("\\s+")).filter { it.isNotBlank() }

        val upperCount = trimmed.count { it.isUpperCase() }
        val letterCount = trimmed.count { it.isLetter() }.coerceAtLeast(1)
        val capsRatio = upperCount.toFloat() / letterCount

        val exclamations = trimmed.count { it == '!' }
        val questionMarks = trimmed.count { it == '?' }
        val hasCurtCommand = lower.startsWith("fix ") || lower.startsWith("do ") ||
            lower.startsWith("send ") || lower.startsWith("stop ") || lower.endsWith(" now.") || lower.endsWith(" now!")

        val aggressiveHits = AGGRESSIVE_TERMS.count { lower.contains(it) }
        val passiveAggressiveHits = PASSIVE_AGGRESSIVE_TERMS.count { lower.contains(it) }
        val defensiveHits = DEFENSIVE_TERMS.count { lower.contains(it) }
        val warmHits = WARM_TERMS.count { lower.contains(it) }
        val assertiveHits = ASSERTIVE_TERMS.count { lower.contains(it) }
        val unclearHits = UNCLEAR_TERMS.count { lower.contains(it) }

        var risk = 0.08f
        risk += aggressiveHits * 0.23f
        risk += passiveAggressiveHits * 0.19f
        risk += defensiveHits * 0.15f
        if (hasCurtCommand && warmHits == 0) risk += 0.18f
        if (exclamations > 1 || (exclamations == 1 && warmHits == 0 && aggressiveHits > 0)) {
            risk += exclamations.coerceAtMost(3) * 0.09f
        }
        if (capsRatio > 0.35f && trimmed.length > 12) risk += 0.28f
        risk -= (warmHits * 0.06f).coerceAtMost(0.12f)
        if (questionMarks > 0 && aggressiveHits == 0 && defensiveHits == 0) risk -= 0.02f
        if (preset == ContextPreset.WORK || preset == ContextPreset.NEGOTIATION) {
            risk += aggressiveHits * 0.05f
        }
        val clampedRisk = risk.coerceIn(0.03f, 0.97f)

        val rawToneScores = floatArrayOf(
            // WARM
            0.42f + warmHits * 0.55f + (if (questionMarks == 1 && aggressiveHits == 0 && assertiveHits == 0) 0.15f else 0f) - aggressiveHits * 0.30f,
            // ASSERTIVE
            0.30f + assertiveHits * 0.50f + (if (words.size >= 8 && aggressiveHits == 0) 0.15f else 0f),
            // DEFENSIVE
            0.12f + defensiveHits * 0.72f,
            // PASSIVE_AGGRESSIVE
            0.10f + passiveAggressiveHits * 0.75f,
            // BLUNT
            0.12f + aggressiveHits * 0.58f + (if (hasCurtCommand && warmHits == 0) 0.48f else 0f) + (if (capsRatio > 0.45f) 0.35f else 0f),
            // UNCLEAR
            0.10f + unclearHits * 0.55f + (if (questionMarks > 1) 0.35f else 0f),
        )
        val toneProbs = ToneSchema.softmax(rawToneScores)
        val toneMap = ToneLabel.entries.mapIndexed { idx, label -> label to toneProbs[idx] }.toMap()
        val primaryTone = toneMap.maxByOrNull { it.value }?.key ?: ToneLabel.WARM

        val clarity = (0.62f + (words.size.coerceAtMost(16) * 0.018f) - (unclearHits * 0.14f) - (if (capsRatio > 0.45f) 0.14f else 0f))
            .coerceIn(0.25f, 0.96f)

        val answeredScore = if (!incomingReply.isNullOrBlank()) {
            val incomingKeywords = incomingReply.lowercase()
                .split(Regex("[^a-z0-9]+"))
                .filter { it.length >= 4 && it !in STOP_WORDS }
                .toSet()
            if (incomingKeywords.isEmpty()) {
                0.72f
            } else {
                val overlap = incomingKeywords.count { lower.contains(it) }
                (0.32f + (overlap.toFloat() / incomingKeywords.size.coerceAtMost(4)) * 0.58f)
                    .coerceIn(0.18f, 0.95f)
            }
        } else {
            null
        }

        val elapsed = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(1L)
        return LayaAnalysis(
            escalationRisk = clampedRisk,
            riskLevel = ToneSchema.classifyRisk(clampedRisk, preset),
            primaryTone = primaryTone,
            toneProbabilities = toneMap,
            clarityScore = clarity,
            answeredQuestionScore = answeredScore,
            inferenceTimeMs = elapsed,
            isPreviewHeuristic = true,
        )
    }

    companion object {
        private const val TAG = "LayaEngine"
        private const val PREFS_NAME = "unsend_prefs"
        private const val KEY_PREVIEW_MODE = "preview_mode_enabled"
        const val DEFAULT_MODEL_FILENAME = "laya.onnx"
        const val DEFAULT_VOCAB_FILENAME = "vocab.txt"
        const val DEFAULT_TOKENIZER_FILENAME = "tokenizer.json"
        const val DEFAULT_MODEL_URL = "https://github.com/Skullmc1/Unsend/releases/download/v1.0.0/laya.onnx"
        const val DEFAULT_TOKENIZER_URL = "https://github.com/Skullmc1/Unsend/releases/download/v1.0.0/tokenizer.json"
        private const val UNLOAD_GRACE_MS = 15_000L

        private const val TEMP_SCORE = 1.25143f
        private const val TEMP_CHOICE_6 = 1.00002f
        private const val TEMP_NOUL = 1.98340f

        private val AGGRESSIVE_TERMS = listOf(
            "ridiculous", "unacceptable", "seriously", "wtf", "whatever", "never",
            "always do this", "waste of time", "stop", "annoying", "stupid", "honestly",
            "makes no sense", "fix it", "shut up", "useless", "wrong", "terrible",
        )
        private val PASSIVE_AGGRESSIVE_TERMS = listOf(
            "per my last", "as i already", "just saying", "fine.", "k.", "thanks in advance",
            "if you actually", "no worries at all", "interesting choice", "good luck with that",
            "sure whatever", "must be nice",
        )
        private val DEFENSIVE_TERMS = listOf(
            "i didn't", "did not", "not my fault", "my fault", "you're the one", "i already told",
            "why are you", "that's not what i", "blame", "had no choice", "wasn't me", "not fair",
        )
        private val WARM_TERMS = listOf(
            "hey", "hi ", "hello", "thanks", "thank you", "appreciate", "please", "happy to",
            "sounds great", "sounds good", "glad", "hope", "sorry", "love", "care",
            "understand", "awesome", "great", "no problem", "for sure", "looking forward",
        )
        private val ASSERTIVE_TERMS = listOf(
            "let's", "we need", "i prefer", "i'd like", "by friday", "moving forward",
            "can we", "could we", "please confirm", "here is", "my availability",
            "let me know", "schedule", "update", "deadline", "review", "plan", "suggest",
        )
        private val UNCLEAR_TERMS = listOf(
            "idk", "maybe", "sort of", "kind of", "stuff", "things", "umm", "huh", "wait what",
        )
        private val STOP_WORDS = setOf(
            "what", "when", "where", "which", "that", "this", "with", "from", "have",
            "your", "about", "would", "could", "should", "there", "their", "they",
        )
    }
}
