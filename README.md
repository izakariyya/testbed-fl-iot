<ins>**Resource Efficient Federated Deep Learning for IoT Security Monitoring**</ins>

![Research](https://img.shields.io/badge/Research-IoT%20Security-blue) ![Federated Learning](https://img.shields.io/badge/AI-Federated%20Learning-green) ![Platform](https://img.shields.io/badge/Platform-Edge%20Computing-orange)

**An experimental IoT edge testbed for investigating resource-efficient federated deep learning for distributed security monitoring.**

This repository provides the implementation associated with the research paper Resource Efficient Federated Deep Learning for IoT Security Monitoring (Zakariyya et al., 2022).

The project investigates how federated deep learning can support IoT security monitoring while reducing computational and communication overhead on resource-constrained edge devices.

**Overview**

Internet of Things (IoT) environments increasingly require intelligent security monitoring to identify malicious network activity and emerging cyber threats.

Traditional centralised machine-learning approaches often require collecting data from distributed devices at a central location. Federated Learning (FL) offers an alternative by enabling model training across distributed devices while keeping their training data local.

However, deploying deep-learning models in resource-constrained IoT environments introduces challenges, particularly in terms of memory consumption, processing time and communication overhead.

This project investigates Resource-Efficient Federated Deep Learning (REFDL) using a practical IoT testbed.

The research explores optimisation techniques for Federated Averaging (FedAvg), including model pruning and simulated micro-batching, to improve the efficiency of distributed deep-learning operations.

The experimental work includes virtual federated workers and a physical edge-computing testbed based on GB-BXBT-2807 devices.


**Objectives**

- Investigate the application of federated deep learning to IoT security.
- Develop and deploy a practical testbed using GB-BXBT-2807 devices.
- Explore distributed model training across connected devices without requiring raw training data to be centrally collected.
- Support the evaluation of model performance and deployment considerations in resource-constrained environments.

**Research Focus**

- Federated learning and distributed deep learning.
- IoT cybersecurity and security monitoring.
- Resource-efficient machine-learning deployment.
- Practical evaluation of learning systems on IoT hardware.
  
**Testbed**

The experimental testbed uses GB-BXBT-2807 devices to investigate the implementation of federated deep learning in an IoT environment. Further implementation details, configuration instructions, and experimental procedures can be documented here to support reproducibility.

**Getting Started**

Clone the repository:

git clone https://github.com/izakariyya/testbed-fl-iot.git
cd testbed-fl-iot


Refer to the project source code and configuration files for implementation details. Additional setup and execution instructions should be added as the environment requirements are documented.

