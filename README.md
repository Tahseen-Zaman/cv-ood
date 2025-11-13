
# **Adversarial & OOD Testing with MNIST using CNN Model**

## **Current Features**

### **Baseline CNN Classifier**

* Small convolutional network trained on MNIST
* Architecture: Conv → MaxPool → Flatten → Dense(64) → Dense(10)
* One-epoch training for quick experimentation
* Serves as a clean baseline to probe vulnerabilities

### **FGSM Adversarial Attack**

* Correct implementation of FGSM using `tf.GradientTape`
* Computes ∂L/∂x and adds ε·sign(gradient) perturbations
* Produces adversarial images that visually resemble originals
* Demonstrates classical misclassification under imperceptible noise
* Includes side-by-side visualization of clean vs adversarial samples

### **OOD Behavior (Fashion-MNIST)**

* Evaluates MNIST-trained model on a Fashion-MNIST input
* Shows high-confidence misclassification on distribution-shifted data
* Illustrates why softmax confidence does not reflect trustworthiness
* Replicates early OOD-detection diagnostics used in robustness research

---


## **TODO: Planned Extensions**

### **Adversarial Attacks**

* PGD (iterative adversarial attack)
* CW-style optimization (loss-based iterative attack)
* Configurable ε-sweeps and perturbation schedules

### **Attribution & Interpretability**

* Integrated Gradients for feature-shift analysis
* Attribution drift between clean, adversarial, and OOD inputs
* Visualization modules for saliency/IG maps

### **Calibration & Reliability**

* Temperature scaling for post-hoc calibration
* Expected Calibration Error (ECE) calculation
* Confidence comparison across clean/adversarial/OOD sets

### **OOD Detection**

* Softmax-entropy thresholds
* Logit-based OOD scoring
* ODIN-style temperature + perturbation approach

---

