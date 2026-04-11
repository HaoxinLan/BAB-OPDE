# BAB-OPDE

## Bayesian Adaptive B-Spline-based Optimal Probability Density Estimation for Mineral Flotation Bubble Size and Morphology Monitoring

[![Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)](https://github.com/YOUR_USERNAME/REPO_NAME)

> **📢 Important Notice**
>
> This paper is currently **under review**.

---

## 📘 Introduction

Froth flotation, the most widely-used mineral separation technique, requires generating mineralized bubbles with appropriate size and stability, necessitating precise monitoring of bubble size and morphological characteristics (BSMCs) distributions. Conventional probability density function (PDF) estimation (PDFE) methods, such as histograms and kernel density estimation (KDE), often lack the interpretability and robustness required for effective process control.  Meanwhile, existing B-spline-based PDFE methods, though suitable for stochastic process monitoring, struggle to achieve optimal PDF results. To overcome the above limitations, this article proposes a **Bayesian Adaptive B-spline-based optimal PDFE (BAB-OPDE)** method for characterizing the distribution of flotation BSMCs. 

### Key Contributions

- BAB-OPDE establishes univariate and multivariate joint PDFs using weighted B-splines, parameterizing the statistical distributions to enhance model intuitiveness and practicality.
- By incorporating a prior smoothing constraint, the PDFE method effectively mitigates noise interference and reduces overfitting risks.
- BAB-OPDE offers the dynamical adjustment strategy of the number of basis splines based on data characteristics, significantly enhancing flexibility and fitting accuracy. 

---

## 📊 Results

### A. Analysis of Validation Experiments based on Numerical Simulation Systems

**Fig. 5** shows that the traditional B-spline method can capture the overall trend of the underlying distribution, but its reconstructed surface exhibits noticeable oscillations and distortions, particularly around the multimodal regions. KDE produces a smoother estimate; however, the multimodal structure is still blurred, and the peak heights and locations deviate from those of the true PDF. In contrast, BAB-OPDE yields a fitted surface that is much closer to the true PDF, with more accurate recovery of the number of modes as well as their shapes and positions, demonstrating its superior capability for two-dimensional joint PDF estimation.

<img width="3130" height="1498" alt="image" src="https://github.com/user-attachments/assets/db38d56d-c404-4e66-b7c2-9c415804733e" />


### B. Convergence Analysis

The convergence behavior under the default setting is illustrated in **Fig. 6**. The figure presents the stage-wise evolution of the objective value, the active basis number, the posterior expectations of the inverse variance parameters, and the norm of the posterior mean coefficient vector.

<img width="1542" height="1670" alt="image" src="https://github.com/user-attachments/assets/2cd51319-e3ab-42e4-a2d8-a384094410c1" />

### C. Case Studies on Copper Flotation Process

**Fig. 14** displays the joint distributions of bubble eccentricity and orientation at key frames. Each small image at these timestamps presents the original froth image, visually illustrating the correlation between feature evolution and morphological changes.
<img width="3349" height="667" alt="image" src="https://github.com/user-attachments/assets/b3731000-b072-4b38-96f1-fe4c76aa1688" />

## 🔐 BAB-OPDE Code Availability

- **After paper acceptance**
  - The complete source code will be **fully released in this repository**

- **During the review stage**
  - Access can be granted **for peer review or academic research purposes only**
  - Contact Email: lhx.contact@foxmail.com

---

## 📌 Notes

- This project is intended **for academic research and non-commercial use only**
- Redistribution of the code or data without permission is prohibited
- The repository will be updated promptly after paper acceptance

Thank you for your interest in our work!
