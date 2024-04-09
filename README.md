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
git clone https://github.com/MinGink/TeamComm.git
cd Teamcomm
conda create -n teamcomm python==3.8
conda activate teamcomm
bash install.sh
```


## Usage Instructions

### Running the Training Process

Execute the following command to initiate the training process:

```bash
python main.py --env your_environment --map your_env_map  --agent teamcomm
```

### Parameter Descriptions

Below are some of the available command-line arguments along with their descriptions:

- `--use_cuda`: Set to `True` to enable CUDA (if available).
- `--env`: Specifies the name of the environment to use.
- `--map`: Specifies the name of the environment map to use.
- `--agent`: Specifies the name of the algorithm to use.
- `--seed`: Set the random seed to ensure the reproducibility of the experiment.
- `--use_offline_wandb`: Set to `True` to enable offline wandb experiment logging.
- `--use_multiprocessing`: Set to `True` to enable multi-process training.
- `--total_epoches`: Set the total number of training epochs.
- `--n_processes`: Set the number of concurrent processes.
- `--att_head`: Set the number of attention heads (applicable to certain algorithms only).
- `--hid_size`: Set the size of the hidden layer (applicable to certain algorithms only).
