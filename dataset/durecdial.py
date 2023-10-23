import json
import pickle
import re
import copy
from dataset.base import Dataset
from config.config import DURECDIAL_TARGET_GOALS


class DuRecdial(Dataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        self.target_goals = DURECDIAL_TARGET_GOALS
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)

    def read_data(self, data_path):
        """Function that reads the Durecdial dataset.
        Returns:
            _type_: list of json strings
        """
        with open(data_path, 'r') as f:
            data = f.readlines()
            assert len(data) > 0
        return data

    def process_data(self, data):
        """method that process the conversations to get input instances.
        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_instances = []
        for conv_id, line in enumerate(data):
            instances = self.construct_instances(conv_id, line)
            all_instances.extend(instances)
        return all_instances

    def repurpose_dataset(self, data):
        """convert the original goal-driven setting to the target-driven CRS setting.
        only consider recommendation-oriented conversations including food, movie, music, poi recommendation

        Args:
            data (_type_): list of json strings, each element is a conversation.

        Returns:
            _type_: list of dictionary each element corresponds to a repurposed conversation,
        """
        new_data = []
        for line in data:
            line = json.loads(line)
            # each line is a conversation
            # in each conversation we have user_profile, goal_sequencs, topics_sequences and conversations.
            scenario = line['goal']
            steps = scenario.split('-->')

            # get the target goal and target topic
            i = len(steps) - 1
            while i >= 0 and ("Say goodbye" in steps[i] or 'recommendation' not in steps[i]):
                i = i - 1

            # we can not find the target recommendation goal
            if i < 0:
                continue

            # preprocessing to get the target goal and the target topic
            target_goal = re.sub(r'\(.*?\)', '', steps[i]).replace(')', '').strip()
            target_topic = steps[i].replace(target_goal, "")[1:-1].strip()

            # there are some cases such as "A. B", B is the accepted item therefore we want to get B.
            if len(target_topic.split('、')) == 2:
                target_topic = target_topic.split('、')[-1].strip()

            target_goal = re.sub(r'[0-9]', '', target_goal).replace("[]", '').strip()
            # if the target goal is not in our considered target list.
            assert target_goal in self.target_goals
            line['target_goal'] = target_goal
            line['target_topic'] = target_topic
            new_data.append(line)
        return new_data

    def construct_instances(self, conv_id, conv):
        """ method that constructs input examples from a conversation
        each instance consists of task background, dialogue context and its corresponding response.

        Args:
            conv_id (_type_): the index of the input conversation
            conv (_type_): the conversation

        Returns:
            _type_: list of input instances.
        """
        instances = []
        task_background = {
            "target_goal": conv['target_goal'],
            "target_topic": conv['target_topic'],
            "user_profile": conv['user_profile'],
        }
        utts = []
        goals = []
        topics = []
        # even for user, and odd for agent.
        role = 0
        if conv['goal_type_list'][0] == "Greetings":
            # agent starts the conversation.
            role = -1
        # print(conv.keys())
        # print(conv['goal_topic_list'])
        for (utt, goal, topic, knowledge) in list(
                zip(conv['conversation'], conv['goal_type_list'], conv['goal_topic_list'], conv['knowledge'])):
            # user responses.
            self.goals.append(goal)
            self.topics.append(topic)
            if role % 2 == 0:
                utts.append({'role': 'user', 'content': utt})
            # the agent starts the conversaiton.
            elif role == -1:
                utts.append({'role': 'assistant', 'content': utt})
                goals.append(goal)
                topics.append(topic)
            # system response
            else:
                # constructing an instance.
                instance = {
                    "conv_id": conv_id,
                    "response": utt,
                    "goal": goal,
                    "topic": topic,
                    "knowledge": knowledge,
                    "pre_goals": copy.deepcopy(goals),
                    "pre_topics": copy.deepcopy(topics),
                    "dialogue_context": copy.deepcopy(utts),
                    "task_background": copy.deepcopy(task_background)
                }
                instances.append(instance)
                utts.append({'role': 'assistant', 'content': utt})
                goals.append(goal)
                topics.append(topic)
            role = role + 1
        return instances


if __name__ == '__main__':
    data_path = 'data/DuRecDial/data/en_train.txt'
    durecdial = DuRecdial(data_path=data_path)
    durecdial.read_data()
    durecdial.repurpose_dataset()
