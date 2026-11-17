# Keep ONNX Runtime JNI classes
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep IME service & SetupActivity
-keep public class com.unsend.keyboard.ime.UnsendImeService
-keep public class com.unsend.keyboard.ui.SetupActivity
