package com.unsend.keyboard.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ToneSchemaAndTokenizerTest {

    @Test
    fun `context preset cycles through all four presets`() {
        var preset = ContextPreset.WORK
        preset = preset.next()
        assertEquals(ContextPreset.PERSONAL, preset)
        preset = preset.next()
        assertEquals(ContextPreset.DATING, preset)
        preset = preset.next()
        assertEquals(ContextPreset.NEGOTIATION, preset)
        preset = preset.next()
        assertEquals(ContextPreset.WORK, preset)
    }

    @Test
    fun `risk classification respects preset thresholds`() {
        // Work is stricter than Personal
        assertEquals(RiskLevel.AMBER, ToneSchema.classifyRisk(0.40f, ContextPreset.WORK))
        assertEquals(RiskLevel.GREEN, ToneSchema.classifyRisk(0.40f, ContextPreset.PERSONAL))
        assertEquals(RiskLevel.RED, ToneSchema.classifyRisk(0.70f, ContextPreset.WORK))
        assertEquals(RiskLevel.AMBER, ToneSchema.classifyRisk(0.70f, ContextPreset.PERSONAL))
    }

    @Test
    fun `softmax probabilities sum to 1`() {
        val probs = ToneSchema.softmax(floatArrayOf(1.0f, 2.0f, 0.5f, -1.0f, 0.0f, 1.5f))
        assertEquals(6, probs.size)
        assertEquals(1.0f, probs.sum(), 1e-4f)
        assertTrue(probs[1] > probs[0])
    }

    @Test
    fun `wordpiece tokenizer produces CLS, subwords, SEP, and attention mask`() {
        val vocab = mapOf(
            "[PAD]" to 0,
            "[UNK]" to 100,
            "[CLS]" to 101,
            "[SEP]" to 102,
            "hello" to 7592,
            "play" to 2377,
            "##ing" to 2075,
        )
        val tokenizer = Tokenizer(vocab = vocab, maxLength = 8)
        val result = tokenizer.tokenize("Hello playing")

        assertEquals(8, result.inputIds.size)
        assertEquals(8, result.attentionMask.size)
        assertEquals(101L, result.inputIds[0]) // [CLS]
        assertEquals(7592L, result.inputIds[1]) // hello
        assertEquals(2377L, result.inputIds[2]) // play
        assertEquals(2075L, result.inputIds[3]) // ##ing
        assertEquals(102L, result.inputIds[4]) // [SEP]
        assertEquals(0L, result.inputIds[5]) // [PAD]

        assertEquals(1L, result.attentionMask[0])
        assertEquals(1L, result.attentionMask[4])
        assertEquals(0L, result.attentionMask[5])
    }
}
