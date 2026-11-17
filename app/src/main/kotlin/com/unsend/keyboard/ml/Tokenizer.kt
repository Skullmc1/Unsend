package com.unsend.keyboard.ml

import org.json.JSONObject
import java.io.File

data class TokenizedInput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
)

/**
 * On-device tokenizer supporting both Hugging Face BPE `tokenizer.json` (for ModernBERT/Laya)
 * and WordPiece `vocab.txt`.
 */
class Tokenizer(
    private val vocab: Map<String, Int> = emptyMap(),
    private val bpeRanks: Map<Pair<String, String>, Int> = emptyMap(),
    private val isBpe: Boolean = false,
    private val maxLength: Int = DEFAULT_MAX_LENGTH,
    val unkId: Int = vocab[UNK_TOKEN] ?: 50280,
    val clsId: Int = vocab[CLS_TOKEN] ?: 50281,
    val sepId: Int = vocab[SEP_TOKEN] ?: 50282,
    val padId: Int = vocab[PAD_TOKEN] ?: 50283,
    val maskId: Int = vocab[MASK_TOKEN] ?: 50284,
) {
    fun isLoaded(): Boolean = vocab.isNotEmpty()

    /**
     * Encodes raw text into token IDs without special tokens (used by Laya's sequence builder).
     */
    fun encode(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()
        if (isBpe) return encodeBpe(text)
        val out = mutableListOf<Int>()
        for (word in basicTokenize(text.lowercase())) {
            out.addAll(wordPieceTokenize(word))
        }
        return out
    }

    fun tokenize(text: String): TokenizedInput {
        val tokens = mutableListOf<Int>()
        tokens.add(clsId)
        for (id in encode(text)) {
            if (tokens.size >= maxLength - 1) break
            tokens.add(id)
        }
        tokens.add(sepId)

        val inputIds = LongArray(maxLength) { padId.toLong() }
        val attentionMask = LongArray(maxLength) { 0L }
        val count = tokens.size.coerceAtMost(maxLength)
        for (i in 0 until count) {
            inputIds[i] = tokens[i].toLong()
            attentionMask[i] = 1L
        }
        return TokenizedInput(inputIds = inputIds, attentionMask = attentionMask)
    }

    private fun encodeBpe(text: String): List<Int> {
        val result = mutableListOf<Int>()
        // Split preserving leading spaces (GPT-2/ModernBERT byte-level BPE style)
        val chunks = BPE_SPLIT_REGEX.findAll(text)
        for (match in chunks) {
            val byteEncoded = toByteLevelToken(match.value)
            val bpeTokens = bpe(byteEncoded)
            for (piece in bpeTokens) {
                result.add(vocab[piece] ?: unkId)
            }
        }
        return result
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1 || bpeRanks.isEmpty()) return listOf(token)
        var word = token.map { it.toString() }
        while (word.size > 1) {
            var bestPair: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val pair = word[i] to word[i + 1]
                val rank = bpeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }
            if (bestPair == null) break
            val next = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == bestPair.first && word[i + 1] == bestPair.second) {
                    next.add(bestPair.first + bestPair.second)
                    i += 2
                } else {
                    next.add(word[i])
                    i += 1
                }
            }
            word = next
        }
        return word
    }

    private fun toByteLevelToken(chunk: String): String {
        val bytes = chunk.toByteArray(Charsets.UTF_8)
        val sb = StringBuilder(bytes.size)
        for (b in bytes) {
            sb.append(BYTE_ENCODER[b.toInt() and 0xFF])
        }
        return sb.toString()
    }

    private fun basicTokenize(text: String): List<String> {
        val result = mutableListOf<String>()
        val current = StringBuilder()
        for (ch in text) {
            when {
                ch.isWhitespace() -> {
                    if (current.isNotEmpty()) {
                        result.add(current.toString())
                        current.clear()
                    }
                }
                isPunctuation(ch) -> {
                    if (current.isNotEmpty()) {
                        result.add(current.toString())
                        current.clear()
                    }
                    result.add(ch.toString())
                }
                else -> current.append(ch)
            }
        }
        if (current.isNotEmpty()) {
            result.add(current.toString())
        }
        return result
    }

    private fun wordPieceTokenize(word: String): List<Int> {
        if (vocab.isEmpty()) {
            return listOf((word.hashCode() and 0x7FFF) + 1000)
        }
        if (word.length > MAX_WORD_CHARS) return listOf(unkId)

        val outputIds = mutableListOf<Int>()
        var start = 0
        var isBad = false

        while (start < word.length) {
            var end = word.length
            var curSubstrId: Int? = null
            while (start < end) {
                val substr = if (start == 0) {
                    word.substring(start, end)
                } else {
                    "##" + word.substring(start, end)
                }
                val id = vocab[substr]
                if (id != null) {
                    curSubstrId = id
                    break
                }
                end--
            }
            if (curSubstrId == null) {
                isBad = true
                break
            }
            outputIds.add(curSubstrId)
            start = end
        }

        return if (isBad) listOf(unkId) else outputIds
    }

    private fun isPunctuation(ch: Char): Boolean {
        val cp = ch.code
        if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)) return true
        return Character.getType(ch).let {
            it == Character.CONNECTOR_PUNCTUATION.toInt() ||
                it == Character.DASH_PUNCTUATION.toInt() ||
                it == Character.START_PUNCTUATION.toInt() ||
                it == Character.END_PUNCTUATION.toInt() ||
                it == Character.OTHER_PUNCTUATION.toInt()
        }
    }

    companion object {
        const val DEFAULT_MAX_LENGTH = 256
        private const val MAX_WORD_CHARS = 100
        private const val PAD_TOKEN = "[PAD]"
        private const val UNK_TOKEN = "[UNK]"
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val MASK_TOKEN = "[MASK]"

        private val BPE_SPLIT_REGEX = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        private val BYTE_ENCODER: CharArray by lazy {
            val bs = mutableListOf<Int>()
            for (b in 33..126) bs.add(b)
            for (b in 161..172) bs.add(b)
            for (b in 174..255) bs.add(b)
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            CharArray(256).apply {
                for (i in bs.indices) {
                    this[bs[i]] = cs[i].toChar()
                }
            }
        }

        fun fromVocabFile(file: File, maxLength: Int = DEFAULT_MAX_LENGTH): Tokenizer {
            if (!file.exists()) return Tokenizer(emptyMap(), maxLength = maxLength)
            val map = HashMap<String, Int>(32_000)
            file.bufferedReader().useLines { lines ->
                var index = 0
                for (line in lines) {
                    val token = line.trimEnd('\r', '\n')
                    if (token.isNotEmpty()) {
                        map[token] = index
                    }
                    index++
                }
            }
            return Tokenizer(
                vocab = map,
                isBpe = false,
                maxLength = maxLength,
                unkId = map[UNK_TOKEN] ?: 100,
                clsId = map[CLS_TOKEN] ?: 101,
                sepId = map[SEP_TOKEN] ?: 102,
                padId = map[PAD_TOKEN] ?: 0,
                maskId = map[MASK_TOKEN] ?: 103,
            )
        }

        fun fromTokenizerJson(file: File, maxLength: Int = DEFAULT_MAX_LENGTH): Tokenizer {
            if (!file.exists()) return Tokenizer(emptyMap(), maxLength = maxLength)
            val root = JSONObject(file.readText())
            val modelObj = root.getJSONObject("model")
            val vocabObj = modelObj.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(vocabObj.length() + 128)
            val keys = vocabObj.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                vocab[key] = vocabObj.getInt(key)
            }

            if (root.has("added_tokens")) {
                val added = root.getJSONArray("added_tokens")
                for (i in 0 until added.length()) {
                    val item = added.getJSONObject(i)
                    vocab[item.getString("content")] = item.getInt("id")
                }
            }

            val mergesMap = HashMap<Pair<String, String>, Int>(64_000)
            if (modelObj.has("merges")) {
                val merges = modelObj.getJSONArray("merges")
                for (i in 0 until merges.length()) {
                    val entry = merges.get(i)
                    when (entry) {
                        is String -> {
                            val parts = entry.split(" ", limit = 2)
                            if (parts.size == 2) mergesMap[parts[0] to parts[1]] = i
                        }
                        is org.json.JSONArray -> {
                            if (entry.length() >= 2) {
                                mergesMap[entry.getString(0) to entry.getString(1)] = i
                            }
                        }
                    }
                }
            }

            return Tokenizer(
                vocab = vocab,
                bpeRanks = mergesMap,
                isBpe = true,
                maxLength = maxLength,
                unkId = vocab[UNK_TOKEN] ?: 50280,
                clsId = vocab[CLS_TOKEN] ?: 50281,
                sepId = vocab[SEP_TOKEN] ?: 50282,
                padId = vocab[PAD_TOKEN] ?: 50283,
                maskId = vocab[MASK_TOKEN] ?: 50284,
            )
        }
    }
}
