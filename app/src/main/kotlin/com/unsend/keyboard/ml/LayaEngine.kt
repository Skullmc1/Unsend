package com.unsend.keyboard.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.SystemClock
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
import java.nio.ByteBuffer
import java.nio.LongBuffer
import java.util.EnumSet

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
                val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
                val options = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(2)
                    runCatching {
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                    }
                }
                ortSession = env.createSession(modelFile.absolutePath, options)
                val vocabFile = findVocabFile()
                tokenizer = when {
                    vocabFile == null -> Tokenizer()
                    vocabFile.name.endsWith(".json", ignoreCase = true) -> Tokenizer.fromTokenizerJson(vocabFile)
                    else -> Tokenizer.fromVocabFile(vocabFile)
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
                return@withContext runOnnxInference(
                    env = env,
                    session = session,
                    draft = draft,
                    incomingReply = incomingReply,
                    preset = preset,
                    startMs = startMs,
                )
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
        val seqLen = tokenized.inputIds.size.toLong()
        val shape = longArrayOf(1L, seqLen)

        val rawLogits: FloatArray = sessionMutex.withLock {
            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(tokenized.inputIds),
                shape,
            )
            val maskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(tokenized.attentionMask),
                shape,
            )

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

        val flatInputIds = LongArray(batch * maxSeqLen) { tokenizer.padId.toLong() }
        val flatAttentionMask = LongArray(batch * maxSeqLen) { 0L }
        val flatMarkerPos = LongArray(batch * maxMarkers) { 0L }
        val flatMarkerMask = ByteArray(batch * maxMarkers) { 0 }
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

        // Pack tightly to actualMaxSeq
        val packedIds = LongArray(batch * actualMaxSeq) { tokenizer.padId.toLong() }
        val packedMask = LongArray(batch * actualMaxSeq) { 0L }
        encodedSequences.forEachIndexed { row, (seq, markers) ->
            val rowOffset = row * actualMaxSeq
            for (i in seq.indices) {
                packedIds[rowOffset + i] = seq[i].toLong()
                packedMask[rowOffset + i] = 1L
            }
            val markerOffset = row * maxMarkers
            for (mIdx in markers.indices) {
                if (mIdx < maxMarkers) {
                    flatMarkerPos[markerOffset + mIdx] = markers[mIdx].toLong()
                    flatMarkerMask[markerOffset + mIdx] = 1
                }
            }
        }

        val logitsRows: Array<FloatArray> = sessionMutex.withLock {
            val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(packedIds), longArrayOf(batch.toLong(), actualMaxSeq.toLong()))
            val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(packedMask), longArrayOf(batch.toLong(), actualMaxSeq.toLong()))
            val mPosTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(flatMarkerPos), longArrayOf(batch.toLong(), maxMarkers.toLong()))
            val mMaskTensor = OnnxTensor.createTensor(env, ByteBuffer.wrap(flatMarkerMask), longArrayOf(batch.toLong(), maxMarkers.toLong()), ai.onnxruntime.OnnxJavaType.BOOL)
            val qtypeTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(qtypes), longArrayOf(batch.toLong()))

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

        val upperCount = trimmed.count { it.isUpperCase() }
        val letterCount = trimmed.count { it.isLetter() }.coerceAtLeast(1)
        val capsRatio = upperCount.toFloat() / letterCount

        val exclamations = trimmed.count { it == '!' }
        val questionMarks = trimmed.count { it == '?' }

        val aggressiveHits = AGGRESSIVE_TERMS.count { lower.contains(it) }
        val passiveAggressiveHits = PASSIVE_AGGRESSIVE_TERMS.count { lower.contains(it) }
        val defensiveHits = DEFENSIVE_TERMS.count { lower.contains(it) }
        val warmHits = WARM_TERMS.count { lower.contains(it) }
        val assertiveHits = ASSERTIVE_TERMS.count { lower.contains(it) }

        var risk = 0.14f
        risk += aggressiveHits * 0.22f
        risk += passiveAggressiveHits * 0.17f
        risk += defensiveHits * 0.13f
        risk += (exclamations.coerceAtMost(3) * 0.08f)
        if (capsRatio > 0.35f && trimmed.length > 18) risk += 0.25f
        risk -= warmHits * 0.09f
        if (preset == ContextPreset.WORK || preset == ContextPreset.NEGOTIATION) {
            risk += aggressiveHits * 0.05f
        }
        val clampedRisk = risk.coerceIn(0.03f, 0.97f)

        val rawToneScores = floatArrayOf(
            // WARM
            0.35f + warmHits * 0.55f - aggressiveHits * 0.25f,
            // ASSERTIVE
            0.25f + assertiveHits * 0.45f + (if (trimmed.length > 40) 0.15f else 0f),
            // DEFENSIVE
            0.15f + defensiveHits * 0.65f,
            // PASSIVE_AGGRESSIVE
            0.12f + passiveAggressiveHits * 0.70f,
            // BLUNT
            0.18f + aggressiveHits * 0.50f + (if (trimmed.length < 32 && warmHits == 0) 0.25f else 0f),
            // UNCLEAR
            0.10f + (if (questionMarks > 1 || trimmed.split(' ').size < 4) 0.35f else 0f),
        )
        val toneProbs = ToneSchema.softmax(rawToneScores)
        val toneMap = ToneLabel.entries.mapIndexed { idx, label -> label to toneProbs[idx] }.toMap()
        val primaryTone = toneMap.maxByOrNull { it.value }?.key ?: ToneLabel.WARM

        val words = trimmed.split(Regex("\\s+")).filter { it.isNotBlank() }
        val clarity = (0.58f + (words.size.coerceAtMost(18) * 0.018f) - (if (capsRatio > 0.4f) 0.15f else 0f))
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
        )
        private val PASSIVE_AGGRESSIVE_TERMS = listOf(
            "per my last", "as i already", "just saying", "fine.", "k.", "thanks in advance",
            "if you actually", "no worries at all", "interesting choice", "good luck with that",
        )
        private val DEFENSIVE_TERMS = listOf(
            "i didn't", "not my fault", "you're the one", "i already told", "why are you",
            "that's not what i", "blame", "i had no choice",
        )
        private val WARM_TERMS = listOf(
            "thanks", "thank you", "appreciate", "please", "happy to", "sounds great",
            "glad", "hope you", "sorry", "love", "care", "understand",
        )
        private val ASSERTIVE_TERMS = listOf(
            "let's", "we need", "i prefer", "i'd like", "by friday", "moving forward",
            "can we", "please confirm", "here is", "my availability",
        )
        private val STOP_WORDS = setOf(
            "what", "when", "where", "which", "that", "this", "with", "from", "have",
            "your", "about", "would", "could", "should", "there", "their", "they",
        )
    }
}
