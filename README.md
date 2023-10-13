# TeamComm: Team-wise Efficient Communication in Multi-Agent Reinforcement Learning

TeamComm is a novel framework as proposed in the paper titled "Team-wise Efficient Communication in Multi-Agent Reinforcement Learning" by Ming Yang et al. This framework aims to enhance the communication efficiency among agents in Multi-Agent Systems (MAS) to foster better collaboration and coordination, especially significant while developing citizen-centric AI solutions.

## Abstract

Effective communication is pivotal for the success of Multi-Agent Systems (MAS) as it enables robust collaboration and coordination among agents. Particularly in the development of citizen-centric AI solutions, there's a need for multi-agent systems to attain specific targets through efficient communication. In the realm of multi-agent reinforcement learning, deciding "whom", "how", and "what" to communicate are critical factors for crafting effective policies. TeamComm introduces a dynamic team reasoning policy, allowing agents to dynamically form teams and adapt their communication partners based on task necessities and environment states in both cooperative or competitive scenarios. It employs heterogeneous communication channels comprising intra- and inter- team channels to facilitate diverse information flow. Lastly, TeamComm applies the information bottleneck principle to optimize communication content, guiding agents to convey relevant and valuable information. The experimental evaluations across three popular environments with seven different scenarios exhibit that TeamComm surpasses existing methods in terms of performance and learning efficiency.

## Keywords
- Reinforcement Learning
- Multi-agent System
- Communication
- Cooperation
- Competition

## Setup

### Prerequisites
- Python 3.7+
- PyTorch 1.5+

### Installation
```bash
git clone https://github.com/your-github-username/teamcomm.git
cd teamcomm
conda create -n teamcomm python==3.8
conda activate teamcomm
python install.py
