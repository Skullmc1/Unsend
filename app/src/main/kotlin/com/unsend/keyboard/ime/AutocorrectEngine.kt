package com.unsend.keyboard.ime

import android.content.Context
import com.unsend.keyboard.R
import kotlin.math.abs
import kotlin.math.min

data class AutocorrectResult(
    val suggestions: List<String>,
    val autoReplace: String?,
)

/**
 * Fast on-device autocorrect and prefix-completion engine using a frequency-ranked
 * English dictionary, contraction/typo map, and Damerau-Levenshtein edit distance.
 */
class AutocorrectEngine(context: Context) {

    private val appContext = context.applicationContext
    private var wordsByRank: List<String> = emptyList()
    private var wordRankMap: Map<String, Int> = emptyMap()
    private var wordsByLength: Map<Int, List<Pair<String, Int>>> = emptyMap()

    @Volatile
    private var isLoaded = false

    fun ensureLoaded() {
        if (isLoaded) return
        synchronized(this) {
            if (isLoaded) return
            val list = ArrayList<String>(3600)
            val rankMap = HashMap<String, Int>(4096)
            val byLen = HashMap<Int, MutableList<Pair<String, Int>>>()

            runCatching {
                appContext.resources.openRawResource(R.raw.words_en).bufferedReader().useLines { lines ->
                    var rank = 0
                    for (raw in lines) {
                        val w = raw.trim()
                        if (w.isNotEmpty() && !rankMap.containsKey(w)) {
                            list.add(w)
                            rankMap[w] = rank
                            byLen.getOrPut(w.length) { mutableListOf() }.add(w to rank)
                            rank++
                        }
                    }
                }
            }

            wordsByRank = list
            wordRankMap = rankMap
            wordsByLength = byLen
            isLoaded = true
        }
    }

    fun suggest(rawWord: String): AutocorrectResult {
        if (!isLoaded) ensureLoaded()
        if (rawWord.length < 2 || !rawWord.all { it.isLetter() || it == '\'' }) {
            return AutocorrectResult(emptyList(), null)
        }

        val lower = rawWord.lowercase()
        val isAllCaps = rawWord.length > 1 && rawWord.all { it.isUpperCase() }
        val isTitleCase = rawWord.first().isUpperCase() && !isAllCaps

        fun applyCase(candidate: String): String = when {
            isAllCaps -> candidate.uppercase()
            isTitleCase -> candidate.replaceFirstChar { it.uppercase() }
            else -> candidate
        }

        // 1. Direct contraction / common typo map
        COMMON_FIXES[lower]?.let { fixed ->
            val cased = if (fixed.contains('\'')) fixed else applyCase(fixed)
            return AutocorrectResult(
                suggestions = listOf(cased, rawWord).distinct(),
                autoReplace = cased,
            )
        }

        // 2. Exact dictionary word -> show word + top 2 prefix completions
        if (wordRankMap.containsKey(lower)) {
            val completions = findPrefixCompletions(lower, limit = 2).map(::applyCase)
            return AutocorrectResult(
                suggestions = (listOf(rawWord) + completions).distinct().take(3),
                autoReplace = null,
            )
        }

        // 3. Unknown word -> Damerau-Levenshtein candidates
        val maxDist = if (lower.length <= 4) 1 else 2
        val candidates = mutableListOf<Triple<String, Int, Int>>() // word, dist, rank

        for (len in (lower.length - maxDist).coerceAtLeast(2)..(lower.length + maxDist)) {
            val bucket = wordsByLength[len] ?: continue
            for ((dictWord, rank) in bucket) {
                if (abs(dictWord.length - lower.length) > maxDist) continue
                val dist = damerauLevenshtein(lower, dictWord, maxDist)
                if (dist <= maxDist) {
                    candidates.add(Triple(dictWord, dist, rank))
                }
            }
        }

        if (candidates.isEmpty()) {
            val prefixFallback = findPrefixCompletions(lower, limit = 2).map(::applyCase)
            return AutocorrectResult(
                suggestions = (listOf(rawWord) + prefixFallback).distinct().take(3),
                autoReplace = null,
            )
        }

        candidates.sortWith(compareBy<Triple<String, Int, Int>> { it.second * 1200 + it.third })
        val topFixes = candidates.take(3).map { applyCase(it.first) }
        val best = topFixes.firstOrNull()

        return AutocorrectResult(
            suggestions = (topFixes + rawWord).distinct().take(3),
            autoReplace = best,
        )
    }

    private fun findPrefixCompletions(prefix: String, limit: Int): List<String> {
        val out = mutableListOf<String>()
        for (w in wordsByRank) {
            if (w.length > prefix.length && w.startsWith(prefix)) {
                out.add(w)
                if (out.size >= limit) break
            }
        }
        return out
    }

    private fun damerauLevenshtein(a: String, b: String, maxDist: Int): Int {
        val m = a.length
        val n = b.length
        if (abs(m - n) > maxDist) return maxDist + 1

        val d = Array(m + 1) { IntArray(n + 1) }
        for (i in 0..m) d[i][0] = i
        for (j in 0..n) d[0][j] = j

        for (i in 1..m) {
            var rowMin = Int.MAX_VALUE
            for (j in 1..n) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                var v = min(
                    min(d[i - 1][j] + 1, d[i][j - 1] + 1),
                    d[i - 1][j - 1] + cost,
                )
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    v = min(v, d[i - 2][j - 2] + 1)
                }
                d[i][j] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > maxDist) return maxDist + 1
        }
        return d[m][n]
    }

    companion object {
        private val COMMON_FIXES = mapOf(
            "im" to "I'm",
            "ive" to "I've",
            "id" to "I'd",
            "dont" to "don't",
            "cant" to "can't",
            "wont" to "won't",
            "didnt" to "didn't",
            "isnt" to "isn't",
            "arent" to "aren't",
            "wasnt" to "wasn't",
            "werent" to "weren't",
            "havent" to "haven't",
            "hasnt" to "hasn't",
            "wouldnt" to "wouldn't",
            "couldnt" to "couldn't",
            "shouldnt" to "shouldn't",
            "youre" to "you're",
            "theyre" to "they're",
            "theres" to "there's",
            "heres" to "here's",
            "whats" to "what's",
            "thats" to "that's",
            "teh" to "the",
            "hte" to "the",
            "adn" to "and",
            "nad" to "and",
            "taht" to "that",
            "wiht" to "with",
            "waht" to "what",
            "recieve" to "receive",
            "seperate" to "separate",
            "definately" to "definitely",
            "becuase" to "because",
            "thier" to "their",
            "wich" to "which",
            "wierd" to "weird",
            "untill" to "until",
            "tommorow" to "tomorrow",
            "occured" to "occurred",
            "acheive" to "achieve",
            "beleive" to "believe",
            "calender" to "calendar",
            "embarass" to "embarrass",
            "existance" to "existence",
            "foriegn" to "foreign",
            "goverment" to "government",
            "independant" to "independent",
            "knowlege" to "knowledge",
            "neccessary" to "necessary",
            "occassion" to "occasion",
            "persistant" to "persistent",
            "prefered" to "preferred",
            "publically" to "publicly",
            "recomend" to "recommend",
            "refered" to "referred",
            "succesful" to "successful",
            "truely" to "truly",
            "alot" to "a lot",
            "u" to "you",
            "ur" to "your",
            "r" to "are",
            "thx" to "thanks",
            "pls" to "please",
            "plz" to "please",
        )
    }
}
