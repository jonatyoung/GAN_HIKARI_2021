# Mitigating Data Imbalance in the ALLFLOWMETER_HIKARI2021 Dataset Using Generative Adversarial Networks to Improve Network Intrusion Detection Performance

## 📘 Project Overview

Intrusion Detection Systems (IDS) are critical for safeguarding modern networks from cyber threats. However, IDS effectiveness is often constrained by **imbalanced datasets**, where benign traffic dominates over attack traffic. For example, the **ALLFLOWMETER\_HIKARI2021 dataset** contains 93.2% benign and only 6.8% attack traffic, leading to biased models that underperform on minority (attack) classes.

This project leverages **Generative Adversarial Networks (GAN)** to address this imbalance by generating high-quality synthetic attack samples. GAN-generated data helps balance the dataset, enabling IDS models to improve detection accuracy and robustness, particularly for rare and minority classes \[Goodfellow et al., 2014; Shahriar et al., 2020; Zhao et al., 2024].

By validating synthetic data through statistical and graphical analysis, this project demonstrates GAN’s potential as a **reliable tool for enhancing IDS performance** in real-world cybersecurity applications.

---

## 💼 Business Understanding

### 📌 Problem Statement

Intrusion Detection Systems (IDS) often struggle with **class imbalance** in real-world datasets. In the **ALLFLOWMETER\_HIKARI2021 dataset**, benign traffic constitutes 93.2% of the data, while attack traffic accounts for only 6.8%. This skew leads to:

* IDS models prioritizing benign traffic while underperforming on attack detection.
* A higher risk of **false negatives**, where actual intrusions are misclassified as normal traffic.
* Limited adaptability of IDS to **emerging and rare cyber threats**, undermining network security.

### 🎯 Objective

This project aims to:

1. **Implement Generative Adversarial Networks (GAN)** to mitigate data imbalance in IDS datasets.
2. **Generate high-quality synthetic attack traffic** to balance class distribution.
3. **Evaluate GAN-generated data** through statistical and graphical validation.
4. **Measure IDS performance improvement** after training on the balanced dataset.

### 💡 Solutions

To achieve the objective, the following solutions are proposed:

* **Data Augmentation with GAN**: Train a custom GAN architecture (NetworkTrafficGAN) on the HIKARI-2021 dataset to generate synthetic attack flows.
* **Balanced Training Dataset**: Combine original benign samples with GAN-generated attack samples to achieve a more balanced distribution.
* **Validation & Benchmarking**: Use distribution comparison, correlation analysis, and Kolmogorov-Smirnov tests to assess the similarity between real and synthetic data.
* **Performance Evaluation**: Compare IDS metrics (precision, recall, F1-score, ROC-AUC) before and after GAN balancing to validate improvements.

---

## 📊 Data Understanding

The dataset used in this project is **ALLFLOWMETER\_HIKARI2021**, derived from the HIKARI-2021 dataset introduced by Ferriyan et al. (2021) \[Ferriyan et al., 2021]. It contains **encrypted synthetic attack traffic** and benign network traffic, making it suitable for evaluating IDS models under realistic conditions.

* **Dataset Source**: HIKARI-2021 (Applied Sciences Journal, MDPI)
* **Original Paper**: [Applied Sciences, Vol. 11(17):7868](https://www.mdpi.com/2076-3417/11/17/7868)
* **Download Dataset**: [Kaggle – ALLFLOWMETER\_HIKARI2021](https://www.kaggle.com/datasets/kk0105/allflowmeter-hikari2021)
* **Classes**:

  * **Benign Traffic**: 517,582 samples (93.2%)
  * **Attack Traffic**: 37,696 samples (6.8%)
* **Features**: Network flow-level features extracted using **all-FlowMeter** (e.g., packet length, duration, inter-arrival times).
* **Challenge**: Severe **class imbalance**, where IDS models tend to classify most traffic as benign, failing to generalize to rare attack patterns.

This imbalance motivates the use of **GAN-based synthetic oversampling**, which generates **realistic and diverse attack flows** to balance the dataset without discarding useful benign samples.

---

## 🛠️ Data Preparation

The dataset preparation pipeline consists of several preprocessing steps to ensure compatibility with GAN training:

1. **Load Dataset**

   * Source: [Kaggle – ALLFLOWMETER\_HIKARI2021](https://www.kaggle.com/datasets/kk0105/allflowmeter-hikari2021)
   * File: `ALLFLOWMETER_HIKARI2021.csv`

2. **Data Cleaning**

   * Remove irrelevant columns: `"Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh"`.
   * Ensure only numerical features remain for GAN input.

3. **One-Hot Encoding**

   * Encode categorical variable `"traffic_category"` into one-hot vectors.

4. **Filter Columns by Data Type**

   * Drop all non-numeric features.
   * Retain only `float64` and `int64` types.

5. **Normalization**

   * Scale features to the range **\[-1, 1]** using MinMaxScaler for stable GAN convergence.

6. **Save Preprocessed Data**

   * Output file: `raw_data_preprocessed.csv`

---

## 🤖 Modeling & Result

### Model Architecture – NetworkTrafficGAN

* **Generator**: Accepts latent vector (`z ∈ R^100`) and outputs synthetic network traffic samples.
* **Discriminator**: Distinguishes between real and synthetic samples.
* **Training Strategy**:

  * Optimizer: Adam (learning rate = 0.0002, β1=0.5)
  * Loss Function: Binary Cross Entropy (BCE)
  * Training Epochs: 500

### Training Results

* **Discriminator Loss (d\_loss)**: Stabilized between **0.91–0.96**, indicating its consistent ability to distinguish real vs. fake data.
* **Generator Loss (g\_loss)**: Decreased from **2.52 → \~1.10**, showing steady improvement in generating realistic samples.
* **Training Dynamics**: Balanced competition between Generator and Discriminator → no mode collapse, suggesting convergence to a stable equilibrium.

### Output

* **Synthetic Dataset**: Generated attack flows to balance the dataset.
* **Result**: IDS models trained on GAN-augmented data showed improved ability to detect minority attack traffic.

---

## 📈 Evaluation

### Statistical Validation

* **Kolmogorov-Smirnov (KS) Test**:

  * For features like `flow_iat.tot` and `active.max`, the KS statistic values were **0.8671** and **0.6726** with **p-value=0.0000**, showing significant distribution differences between original and synthetic data.
* **Descriptive Statistics**:

  * Some features exhibited high deviations in mean and standard deviation (up to **-95% difference**), highlighting areas for further GAN refinement.

### Visual Validation

* Feature distributions (histograms) and correlation heatmaps revealed that GAN-generated data partially followed the real data’s patterns but failed to fully replicate long-tail distributions.

### Model Performance

* **Improvement**: IDS models showed **better recall and F1-score for minority (attack) classes** after GAN augmentation compared to baseline.
* **Limitation**: Despite performance gains, discrepancies in distribution similarity suggest GAN outputs may not fully generalize to highly complex attack behaviors.

### **Answering Business Understanding**

* **Problem**: Traditional IDS models struggle with imbalanced datasets, under-detecting minority attacks.
* **Objective**: Improve IDS performance by generating realistic synthetic attack data using GAN.
* **Solution**: NetworkTrafficGAN successfully created synthetic data that improved IDS recall and robustness, though further tuning is needed to fully match original distributions.

---

## 📝 Conclusion

In this experiment, we addressed the severe class imbalance in the HIKARI-2021 dataset, where benign traffic dominated (93.2%) while attack traffic represented only a small fraction (6.8%). To mitigate this imbalance, we employed a **Generative Adversarial Network (GAN)** to synthesize realistic attack traffic samples, thereby increasing the representation of minority classes.

The results demonstrated that balancing the dataset using GAN significantly improved the performance of the Intrusion Detection System (IDS) model. Compared to training on the raw imbalanced dataset, the GAN-augmented dataset yielded:

* **Higher recall and F1-score on attack classes**, indicating better detection of malicious traffic.
* **Reduced bias towards benign traffic**, leading to a more reliable and robust IDS.
* **Overall improved generalization**, as the model could detect rare attack patterns that were previously underrepresented.

This study highlights the effectiveness of GAN-based data augmentation in enhancing IDS performance on imbalanced network datasets. Future work may explore the integration of other advanced oversampling methods, hybrid deep learning models, and evaluation on additional real-world datasets to further validate robustness.

---

## 🔗 References

* Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NIPS. [Link](https://papers.nips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
* Shahriar, M.H., et al. (2020). *G-IDS: Generative Adversarial Networks Assisted Intrusion Detection System*. arXiv:2006.00676. [Link](https://arxiv.org/abs/2006.00676)
* Zhao, X., et al. (2024). *Enhancing Network Intrusion Detection Performance using GAN*. arXiv:2404.07464. [Link](https://arxiv.org/html/2404.07464v1)
* Ferriyan, A., et al. (2021). *Generating Network Intrusion Detection Dataset Based on Real and Encrypted Synthetic Attack Traffic*. Applied Sciences, 11(17), 7868. DOI: [10.3390/app11177868](https://www.mdpi.com/2076-3417/11/17/7868)
