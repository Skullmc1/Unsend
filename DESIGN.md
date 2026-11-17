# Unsend — On-Device Tone & Risk Keyboard (Android)

**Unsend** is a 100% offline, privacy-first Android custom keyboard (IME) powered by **Laya**, a single-forward-pass decision model. Instead of sending keystrokes to the cloud or rewriting your messages with generative AI, Unsend acts as a real-time mirror above your keyboard—flagging tone, escalation risk, and clarity before you hit send.

---

## 1. Core Value Proposition
* **Zero Cloud, Zero Keystroke Logging:** All inference runs locally on-device using ONNX Runtime Android. Works in airplane mode.
* **Fast & Battery-Friendly:** Laya evaluates multiple questions in a **single forward pass** rather than generating tokens one by one.
* **Authentic Human Writing:** Acts as a speedometer/gauge ("Mirror, not a writer")—you stay in control of your own words.

---

## 2. Target Platform & Tech Stack
* **Platform:** Android only (API 26+ / Android 8.0+ recommended for NNAPI support).
* **Language:** Kotlin.
* **Keyboard Engine:** Android `InputMethodService` (full system-wide custom keyboard, similar to SwiftKey/Gboard).
* **ML Runtime:** `com.microsoft.onnxruntime:onnxruntime-android` (with NNAPI / XNNPACK execution providers).
* **Async / Concurrency:** Kotlin Coroutines (`Dispatchers.Default` for tokenization & inference, debounced on input).

---

## 3. Keyboard UI & HUD Layout

### A. The Unsend Toolbar (~48dp pinned above keys)
Because vertical screen space is tight when a keyboard is open, the HUD stays compact:
1. **Context Preset Pill:** Tap to cycle `[ Work | Personal | Dating | Negotiation ]` (adjusts risk/tone framing).
2. **Live Status & Tone Tag:** Color-coded indicator + primary signal (e.g., `🟢 Clear & Warm`, `🟡 68% Defensive`, `🔴 High Escalation`).
3. **Reply-Context Chip (`📋 Check Reply`):** Optional button that reads the last copied incoming message from the clipboard so Laya can compare your draft against what the other person actually said.
4. **Expand Drawer Button:** Tapping the status bar opens a compact overlay above the keys with full probability meters.

### B. Base Keyboard (MVP)
* Standard QWERTY layout + Shift, Backspace, Space, Enter.
* Numbers & Symbols layer (`?123`).
* Clean, distraction-free key styling with haptic feedback.
* *(Future iterations: swipe typing, autocorrect/word suggestions, emoji picker.)*

---

## 4. Laya Decision Schema (Single-Pass Evaluation)

Unsend reads the current draft text from `InputConnection` (via `getExtractedText` / `getTextBeforeCursor` + `getTextAfterCursor`) and optionally the pasted incoming message.

In a single forward pass, Laya scores:

| Signal | Question Type | Output |
| :--- | :--- | :--- |
| **Escalation / Regret Risk** | Scalar / Probability | `0.0 – 1.0` (Mapped to Green / Amber / Red) |
| **Primary Tone** | Multiple Choice | `Warm`, `Assertive`, `Defensive`, `Passive-Aggressive`, `Blunt`, `Unclear` |
| **Clarity** | Yes / No | `P(clear)` |
| **Answered Question?** *(Reply Mode)* | Yes / No | `P(answers_incoming)` |

---

## 5. Inference & Performance Strategy

* **Never Block the Main Thread:** Typing must remain 60/120fps. Tokenization and ONNX inference run strictly on background coroutines.
* **Debounced Triggering:** Do **not** run Laya on every keystroke. Trigger inference only when:
  * User pauses typing for **~700ms**, OR
  * User types sentence-ending punctuation (`.`, `?`, `!`), OR
  * User manually taps the toolbar status pill.
* **Cancellation:** If the user resumes typing mid-inference, cancel/discard stale results using a `Job` or `StateFlow` sequence ID.
* **Minimum Length Gate:** Skip inference if the draft is under ~15–20 characters (e.g., `"ok"`, `"sounds good"`).

---

## 6. Model Packaging & Storage
* **Quantization:** Use an `int8` / `q4` quantized ONNX export of Laya (~200–420 MB) to fit comfortably in Android memory.
* **Delivery:** Do not bundle the large `.onnx` file directly inside the base APK. Instead:
  * The **Companion Setup Activity** downloads (or imports) the model + tokenizer files into `context.filesDir/models/` on first launch.
  * The `InputMethodService` checks for model availability on startup; if missing, the toolbar displays a `"Finish setup in Unsend"` prompt while still functioning as a normal keyboard.
* **Memory Management:** Set `android:largeHeap="true"` in `AndroidManifest.xml` if needed, and lazily load/unload the `OrtSession` when `onStartInputView()` / `onFinishInputView()` fires (with a short grace timeout so quick app-switching doesn't reload the model repeatedly).

---

## 7. Proposed Project Structure

```text
Unsend/
├── app/
│   ├── src/main/
│   │   ├── AndroidManifest.xml          # Declares InputMethodService + SetupActivity
│   │   ├── res/xml/method.xml           # IME metadata & subtypes
│   │   └── java/com/unsend/keyboard/
│   │       ├── ime/
│   │       │   ├── UnsendImeService.kt  # InputMethodService lifecycle & InputConnection
│   │       │   ├── KeyboardView.kt      # QWERTY / Symbols rendering & key handling
│   │       │   └── HudBarView.kt        # Top toolbar + expandable breakdown drawer
│   │       ├── ml/
│   │       │   ├── LayaEngine.kt        # ONNX Runtime session & inference runner
│   │       │   ├── Tokenizer.kt         # Fast Kotlin/JNI WordPiece/BPE tokenizer
│   │       │   └── ToneSchema.kt        # Question prompts & probability thresholds
│   │       └── ui/
│   │           └── SetupActivity.kt     # Onboarding, Enable IME prompt, Model download
└── DESIGN.md
```

---

## 8. Build Milestones

1. **Milestone 1 — Basic Working IME:** `SetupActivity` + `UnsendImeService` with working QWERTY/Symbols typing and `HudBarView` placeholder.
2. **Milestone 2 — ONNX Runtime Integration:** Load quantized Laya `.onnx` model + tokenizer in `LayaEngine.kt` and run debounced background inference on `InputConnection` text.
3. **Milestone 3 — Live HUD & Reply Mode:** Wire `ToneSchema` probabilities to the toolbar color meter, expandable drawer, context presets, and `📋 Check Reply` clipboard comparison.
4. **Milestone 4 — Polish & Tuning:** Battery/memory profiling, `OrtSession` lifecycle caching, and threshold tuning across Work/Personal/Dating presets.
