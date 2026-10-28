# FSDETR: Frequency-Spatial Feature Enhancement for Small Object Detection
**Jianchao Huang, Fengming Zhang, Haibo Zhu and Tao Yan\***

> Abstract—Small object detection remains a significant challenge due to feature degradation from downsampling, mutual occlusion in dense clusters, and complex background interference. To address these issues, this paper proposes FSDETR, a frequency–spatial feature enhancement framework built upon the RT-DETR baseline. By establishing a collaborative modeling mechanism, the method effectively leverages complementary structural information. Specifically, a Spatial Hierarchical Attention Block (SHAB) captures both local details and global dependencies to strengthen semantic representation. Furthermore, to mitigate occlusion in dense scenes, the Deformable Attention-based Intra-scale Feature Interaction (DA-AIFI) focuses on informative regions via dynamic sampling. Finally, the Frequency-Spatial Feature Pyramid Network (FSFPN) integrates frequency filtering with spatial edge extraction via the Crossdomain Frequency-Spatial Block (CFSB) to preserve fine-grained details. Experimental results show that with only 14.7M parameters, FSDETR achieves 13.9% APS on VisDrone 2019 and 48.95% AP50 tiny on TinyPerson, showing strong performance on small-object benchmarks. 

## Proceeding
This paper has been accepted by IJCNN 2026. The paper link is: https://arxiv.org/abs/2604.14884 and https://ieeexplore.ieee.org/document/11729161/

@INPROCEEDINGS{11729161,
  author={Huang, Jianchao and Zhang, Fengming and Zhu, Haibo and Yan, Tao},
  booktitle={2026 International Joint Conference on Neural Networks (IJCNN)}, 
  title={FSDETR: Frequency-Spatial Feature Enhancement for Small Object Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Printing;Object detection;Frequency;Conferences;Computers;Design methodology;Signal detection;Computer vision;Distance measurement;Small object detection;Feature pyramid;Frequency–spatial feature enhancement;RT-DETR},
  doi={10.1109/IJCNN69854.2026.11729161}}



## Network Architecture
![architecture](images/architecture.png)


## Quantitative Comparisons
![QuantitativeComparisons](images/QuantitativeComparisons.png)

## APs Comparisons

![APsComparisons](images/APsComparisons.png)
