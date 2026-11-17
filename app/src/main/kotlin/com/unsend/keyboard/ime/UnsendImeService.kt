package com.unsend.keyboard.ime

import android.content.ClipboardManager
import android.content.Intent
import android.inputmethodservice.InputMethodService
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.ExtractedTextRequest
import android.widget.LinearLayout
import android.widget.Toast
import com.unsend.keyboard.ml.ContextPreset
import com.unsend.keyboard.ml.LayaAnalysis
import com.unsend.keyboard.ml.LayaEngine
import com.unsend.keyboard.ml.ModelStatus
import com.unsend.keyboard.ml.ToneSchema
import com.unsend.keyboard.ui.SetupActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Unsend system-wide InputMethodService.
 * Combines a distraction-free QWERTY/Symbols keyboard with Laya's single-pass tone & risk mirror.
 */
class UnsendImeService : InputMethodService() {

    private lateinit var layaEngine: LayaEngine
    private var hudBarView: HudBarView? = null
    private var keyboardView: KeyboardView? = null

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var inferenceJob: Job? = null
    private var inferenceSequence: Long = 0L

    private var currentPreset: ContextPreset = ContextPreset.WORK
    private var incomingReplyContext: String? = null
    private var lastAnalysis: LayaAnalysis? = null

    override fun onCreate() {
        super.onCreate()
        layaEngine = LayaEngine(this)
    }

    override fun onCreateInputView(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val hud = HudBarView(this).apply {
            updatePreset(currentPreset)
            onCyclePreset = {
                currentPreset = currentPreset.next()
                updatePreset(currentPreset)
                scheduleEvaluation(immediate = true)
            }
            onStatusClick = {
                if (lastAnalysis != null) {
                    toggleDrawer()
                } else {
                    scheduleEvaluation(immediate = true)
                }
            }
            onToggleReplyCheck = { handleReplyToggle() }
            onClearReplyContext = {
                incomingReplyContext = null
                updateReplyContext(null)
                scheduleEvaluation(immediate = true)
            }
            onOpenSetup = {
                val intent = Intent(this@UnsendImeService, SetupActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                startActivity(intent)
            }
        }

        val keyboard = KeyboardView(this).apply {
            listener = object : KeyboardView.Listener {
                override fun onCharacter(text: String) {
                    val ic = currentInputConnection ?: return
                    ic.commitText(text, 1)
                    val isSentenceEnd = text in SENTENCE_END_PUNCTUATION
                    scheduleEvaluation(immediate = isSentenceEnd)
                }

                override fun onBackspace() {
                    val ic = currentInputConnection ?: return
                    val selected = ic.getSelectedText(0)
                    if (!selected.isNullOrEmpty()) {
                        ic.commitText("", 1)
                    } else {
                        ic.deleteSurroundingText(1, 0)
                    }
                    scheduleEvaluation(immediate = false)
                }

                override fun onEnter() {
                    val ic = currentInputConnection ?: return
                    val info = currentInputEditorInfo
                    val action = (info?.imeOptions ?: 0) and EditorInfo.IME_MASK_ACTION
                    val noEnterAction = ((info?.imeOptions ?: 0) and EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0
                    if (!noEnterAction && action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED) {
                        ic.performEditorAction(action)
                    } else {
                        sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
                    }
                    scheduleEvaluation(immediate = true)
                }
            }
        }

        hudBarView = hud
        keyboardView = keyboard

        root.addView(hud)
        root.addView(keyboard)
        return root
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        if (info != null) {
            keyboardView?.updateImeOptions(info.imeOptions)
        }
        serviceScope.launch {
            layaEngine.ensureLoaded()
        }
        refreshHudIdleOrEvaluate()
    }

    override fun onUpdateSelection(
        oldSelStart: Int,
        oldSelEnd: Int,
        newSelStart: Int,
        newSelEnd: Int,
        candidatesStart: Int,
        candidatesEnd: Int,
    ) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        val draft = readCurrentDraft()
        if (draft.length < ToneSchema.MIN_DRAFT_LENGTH && lastAnalysis != null) {
            lastAnalysis = null
            hudBarView?.renderState(
                modelStatus = layaEngine.getStatus(),
                analysis = null,
                isEvaluating = false,
                draftLength = draft.length,
            )
        }
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        inferenceJob?.cancel()
        layaEngine.scheduleUnload()
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
    }

    private fun handleReplyToggle() {
        if (!incomingReplyContext.isNullOrBlank()) {
            incomingReplyContext = null
            hudBarView?.updateReplyContext(null)
            scheduleEvaluation(immediate = true)
            return
        }

        val clipboard = getSystemService(CLIPBOARD_SERVICE) as? ClipboardManager
        val clipText = clipboard?.primaryClip
            ?.takeIf { it.itemCount > 0 }
            ?.getItemAt(0)
            ?.coerceToText(this)
            ?.toString()
            ?.trim()

        if (clipText.isNullOrBlank()) {
            Toast.makeText(this, "Copy an incoming message first to check your reply.", Toast.LENGTH_SHORT).show()
            return
        }

        incomingReplyContext = clipText
        hudBarView?.updateReplyContext(clipText)
        scheduleEvaluation(immediate = true)
    }

    private fun refreshHudIdleOrEvaluate() {
        val status = layaEngine.getStatus()
        val draft = readCurrentDraft()
        if (status == ModelStatus.NOT_CONFIGURED || draft.length < ToneSchema.MIN_DRAFT_LENGTH) {
            lastAnalysis = null
            hudBarView?.renderState(
                modelStatus = status,
                analysis = null,
                isEvaluating = false,
                draftLength = draft.length,
            )
        } else {
            scheduleEvaluation(immediate = true)
        }
    }

    private fun scheduleEvaluation(immediate: Boolean) {
        val status = layaEngine.getStatus()
        val draft = readCurrentDraft()

        if (status == ModelStatus.NOT_CONFIGURED) {
            inferenceJob?.cancel()
            hudBarView?.renderState(status, null, isEvaluating = false, draftLength = draft.length)
            return
        }

        if (draft.length < ToneSchema.MIN_DRAFT_LENGTH) {
            inferenceJob?.cancel()
            lastAnalysis = null
            hudBarView?.renderState(status, null, isEvaluating = false, draftLength = draft.length)
            return
        }

        inferenceJob?.cancel()
        val seqId = ++inferenceSequence
        val presetSnapshot = currentPreset
        val replySnapshot = incomingReplyContext

        inferenceJob = serviceScope.launch {
            if (!immediate) {
                delay(DEBOUNCE_MS)
            }
            if (seqId != inferenceSequence) return@launch

            hudBarView?.renderState(status, lastAnalysis, isEvaluating = true, draftLength = draft.length)

            val result = runCatching {
                layaEngine.analyze(
                    draft = draft,
                    incomingReply = replySnapshot,
                    preset = presetSnapshot,
                )
            }.getOrNull()

            if (seqId != inferenceSequence) return@launch
            if (result != null) {
                lastAnalysis = result
                hudBarView?.renderState(status, result, isEvaluating = false, draftLength = draft.length)
            } else {
                hudBarView?.renderState(status, lastAnalysis, isEvaluating = false, draftLength = draft.length)
            }
        }
    }

    private fun readCurrentDraft(): String {
        val ic = currentInputConnection ?: return ""
        val extracted = ic.getExtractedText(ExtractedTextRequest(), 0)?.text?.toString()
        if (!extracted.isNullOrEmpty()) return extracted

        val before = ic.getTextBeforeCursor(2000, 0)?.toString().orEmpty()
        val after = ic.getTextAfterCursor(500, 0)?.toString().orEmpty()
        return before + after
    }

    companion object {
        private const val DEBOUNCE_MS = 700L
        private val SENTENCE_END_PUNCTUATION = setOf(".", "?", "!")
    }
}
