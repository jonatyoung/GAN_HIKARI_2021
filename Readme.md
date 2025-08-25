# Mitigating Data Imbalance in the ALLFLOWMETER_HIKARI2021 Dataset Using Generative Adversarial Networks to Improve Network Intrusion Detection Performance

## 📘 Project Overview

Intrusion Detection Systems (IDS) are critical for safeguarding modern networks from cyber threats. However, IDS effectiveness is often constrained by **imbalanced datasets**, where benign traffic dominates over attack traffic. For example, the **ALLFLOWMETER\_HIKARI2021 dataset** contains 93.2% benign and only 6.8% attack traffic, leading to biased models that underperform on minority (attack) classes.

This project leverages **Generative Adversarial Networks (GAN)** to address this imbalance by generating high-quality synthetic attack samples. GAN-generated data helps balance the dataset, enabling IDS models to improve detection accuracy and robustness, particularly for rare and minority classes \[Goodfellow et al., 2014; Shahriar et al., 2020; Zhao et al., 2024].

By validating synthetic data through statistical and graphical analysis, this project demonstrates GAN’s potential as a **reliable tool for enhancing IDS performance** in real-world cybersecurity applications.

---

## 💼 Business Understanding

Cyberattacks continue to evolve in scale and sophistication, causing significant **financial, operational, and reputational risks** for organizations worldwide. Traditional IDS models often fail to detect minority attack patterns due to skewed training data, leaving networks vulnerable to novel or underrepresented threats.

The business needs addressed in this project are:

* **Enhanced Threat Detection**: Improve IDS ability to detect minority attack traffic, reducing the risk of undetected intrusions.
* **Data-Driven Security**: Demonstrate how GAN-generated synthetic data can supplement scarce attack samples and provide balanced datasets for model training.
* **Operational Reliability**: Support organizations in building more **robust IDS pipelines** that can adapt to emerging cybersecurity challenges with minimal manual dataset augmentation.

In essence, this project contributes to the **development of resilient cybersecurity solutions**, reducing potential losses from cyberattacks while ensuring better compliance with data protection standards.

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

## 🔗 References

* Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NIPS. [Link](https://papers.nips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
* Shahriar, M.H., et al. (2020). *G-IDS: Generative Adversarial Networks Assisted Intrusion Detection System*. arXiv:2006.00676. [Link](https://arxiv.org/abs/2006.00676)
* Zhao, X., et al. (2024). *Enhancing Network Intrusion Detection Performance using GAN*. arXiv:2404.07464. [Link](https://arxiv.org/html/2404.07464v1)
* Ferriyan, A., et al. (2021). *Generating Network Intrusion Detection Dataset Based on Real and Encrypted Synthetic Attack Traffic*. Applied Sciences, 11(17), 7868. DOI: [10.3390/app11177868](https://www.mdpi.com/2076-3417/11/17/7868)

---

Do you also want me to **add a GAN-based workflow diagram (Dataset → GAN Balancing → IDS Training → Evaluation)** in Markdown with an image placeholder, so it looks neat on your README?
