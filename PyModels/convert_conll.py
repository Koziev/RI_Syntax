# -*- coding: utf-8 -*-

from __future__ import print_function
import collections
import numpy as np
import codecs
import gym
import gym.spaces
import random
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent


from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import itertools

# -------------------------------------------------------------

class SentenceToken(object):
    '''
    Один токен в корпусном предложении: само слово с порядковым номером
    в предложении (нумерация с 1) и порядковый номер родителя.
    '''
    def __init__(self, index, word, parent_index, tagset_id):
        self.index = index
        self.word = word
        self.parent_index = parent_index
        self.tagset_id = tagset_id

    def is_root(self):
        return self.parent_index == 0

    def get_edge(self):
        return (self.parent_index, self.index)

    def get_parent_index(self):
        return self.parent_index

    def get_word(self):
        return self.word

    def get_tagset_id(self):
        return self.tagset_id

    def get_token_index(self):
        return self.index

    def __str__(self):
        if self.is_root():
            return u'{}:{} ROOT'.format(self.index, self.word)
        else:
            return u'{}:{} --> {}'.format(self.index, self.word, self.parent_index)



class SentenceTokens(object):
    def __init__(self):
        self.tokens = []

    def append_token(self, index, word, parent_index, tagset_id):
        self.tokens.append( SentenceToken(index, word, parent_index, tagset_id) )



class TrainingEpisode(object):
    def __init__(self):
        self.steps = []

    def append_step(self, token_index, word, word_vector, parent_index):
        self.steps.append( (token_index, word, word_vector, parent_index) )

    def get_steps(self):
        return self.steps

    def get_length(self):
        return len(self.steps)

    def get_edges(self):
        edges = []
        for step in self.steps:
            edge = (step[3],step[0]) # (родитель,ребенок)
            edges.append(edge)
        return edges


class TreeBank(object):

    def __init__(self):
        self.max_suffix_len = 4
        pass

    def vectorize_word(self, word, tagset_id):
        X = np.zeros( (self.word_vector_len), dtype='bool' )

        # кодируем char surface
        for i,c in enumerate(word[::-1][:self.max_suffix_len]):
            X[ i*self.bits_per_char + self.char2index[c] ] = True

        # добавляем биты морфологических тегов
        n0 = self.max_suffix_len*self.bits_per_char
        tags = self.id2tagset[tagset_id]
        v = np.zeros( len(self.tag2id) )
        for tag in tags:
            X[ n0 + self.tag2id[tag] ] = True

        return X

    def build_episode(self, sentence):
        episode = TrainingEpisode()
        for token in sentence.tokens:
            word_data = self.vectorize_word(token.get_word(), token.get_tagset_id())
            episode.append_step( token.get_token_index(), token.get_word(), word_data, token.get_parent_index())
        return episode

    def load_conll(self, path):
        self.sentences = []
        self.max_word_len = -1
        self.all_chars = set()
        self.tagset2id = dict()
        self.id2tagset = dict()
        with codecs.open(path, 'r', 'utf-8') as rdr:
            cur_sentence = None
            sentences = []
            line_count = 0
            for line0 in rdr:
                if line0.startswith(u'#'): continue
                line_count += 1
                line = line0.strip()
                if len(line)==0:
                    self.sentences.append(cur_sentence)
                    cur_sentence = None
                else:
                    fields = line.split(u'\t')
                    if u'.' not in fields[0]:
                        token_num = int(fields[0])
                        word = fields[1].lower()
                        parent_num = int(fields[6])
                        part_of_speech = fields[3]
                        tags = fields[5]
                        if cur_sentence is None:
                            cur_sentence = SentenceTokens()

                        tagset = part_of_speech+'|'+tags
                        if tagset not in self.tagset2id:
                            id_tagset = len(self.tagset2id)
                            self.tagset2id[tagset] = id_tagset
                            self.id2tagset[id_tagset] = tagset.split('|')
                        else:
                            id_tagset = self.tagset2id[tagset]

                        cur_sentence.append_token( token_num, word, parent_num, id_tagset )
                        self.max_word_len = max(self.max_word_len, len(word))
                        self.all_chars.update(word)

        self.char2index = dict([ (c,i) for i,c in enumerate(self.all_chars) ])
        self.bits_per_char = len(self.char2index)

        all_tags = set( itertools.chain( *[ tagset.split('|') for tagset in self.tagset2id.keys() ] ) )
        self.tag2id = dict( [ (t,i) for i,t in enumerate(all_tags) ] )

        self.word_vector_len = self.bits_per_char * min( self.max_suffix_len, self.max_word_len ) +\
                               len(self.tag2id)

        self.episodes = [ self.build_episode(sentence) for sentence in self.sentences ]
        print('{} episodes loaded.'.format(len(self.episodes)))
        print('bits_per_chars={}'.format(self.bits_per_char))
        print('word_vector_len={}'.format(self.get_word_vector_len()))

    def choose_episode(self):
        '''
        Случайный выбор нового эпизода.
        :return: случайно выбранный эпизод (объект класса TrainingEpisode).
        '''
        return random.choice(self.episodes)

    def get_word_vector_len(self):
        return self.word_vector_len

# ---------------------------------------------------------------------------

treebank = TreeBank()
treebank.load_conll('../treebank/russian/ru_syntagrus-ud-train.conllu')

# ---------------------------------------------------------------------------


class SyntaxNode(object):
    def __init__(self, token_index, word_str, word_vector):
        self.token_index = token_index
        self.word_str = word_str
        self.word_vector = word_vector
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_head_vector(self):
        return self.word_vector

    def get_token_index(self):
        return self.token_index

    def count_children(self):
        return len(self.children)

    def get_edges(self):
        edges = []
        self.get_children_edges(edges)
        return edges

    def get_children_edges(self, edges):
        for child in self.children:
            edge = (self.token_index, child.token_index) # (родитель,ребенок)
            edges.append(edge)
            child.get_children_edges(edges)



class EpisodeProcessor(object):
    '''
    Вспомогательный класс для обработки действий в рамках одного
    эпизода.
    Хранит исходный поток слов и стек. Каждый элемент входного
    потока и стека представляет из себя фрагмент синтаксического дерева.
    '''
    def __init__(self, episode):
        self.episode = episode
        self.actions_history = []
        self.reward_history = []
        self.punishment = 0.0 # штрафы за недопустимые действия
        self.reward = 0.0
        self.stream = collections.deque( [ SyntaxNode(step[0], step[1], step[2]) for step in episode.get_steps() ] )
        self.stack = []
        self.word_vector_len = len(episode.get_steps()[0][2])
        self.zero_vector = np.zeros( (self.word_vector_len), dtype='bool' )
        self.one1_vector = np.ones( (1), dtype='bool' )
        self.zero1_vector = np.zeros( (1), dtype='bool' )
        self.true_edges = set(self.episode.get_edges())
        self.finished = False


    def is_finished(self):
        return self.finished or\
               (len(self.stream)==1 and len(self.stack)==0) or\
               len(self.actions_history)>self.episode.get_length()*4 # остался только токен-корень

    def perform_action(self, action_code):
        self.actions_history.append(action_code)
        reward = 0.0
        if action_code == 0:
            reward = self.push()
        elif action_code == 1:
            reward = self.pop_child()
        elif action_code == 2:
            reward = self.pop_parent()
        else:
            raise NotImplemented()

        self.reward_history.append( reward )
        return reward

    def push(self):
        # текущий токен из потока вталкиваем на вершину стека
        if len(self.stream)>0:
            cur_word = self.stream[0]
            self.stack.insert(0,cur_word)
            self.stream.popleft()
            return 0.0 # нейтральное действие
        else:
            #self.finished = True
            return -2.0  # недопустимое действие

    def pop_child(self):
        # берем вершину стека и прикрепляем как ветку к текущему токену в потоке
        if len(self.stack) > 0 and len(self.stream) > 0:
            cur_word = self.stream[0]
            tos = self.stack.pop()
            cur_word.add_child(tos)

            parent_index = cur_word.get_token_index()
            child_index = tos.get_token_index()

            edge = (parent_index, child_index)

            if edge in self.true_edges:
                return 1.0 # верное ребро
            else:
                return -1.0 # некорректное ребро
        else:
            #self.finished = True
            return -2.0 # недопустимое действие

    def pop_parent(self):
        # берем вершину стека и прикрепляем как корень к текущему токену в потоке
        if len(self.stack) > 0 and len(self.stream) > 0:
            cur_word = self.stream[0]
            tos = self.stack.pop()
            tos.add_child(cur_word)
            self.stream[0] = tos

            child_index = cur_word.get_token_index()
            parent_index = tos.get_token_index()

            edge = (parent_index, child_index)

            if edge in self.true_edges:
                return 1.0 # верное ребро
            else:
                return -1.0 # некорректное ребро
        else:
            #self.finished = True
            return -2.0 # недопустимое действие

    def get_state(self):
        vectors = []

        # текущее слово
        nb_children = 0
        if len(self.stream)>0:
            cur_word = self.stream[0]
            vectors.append( cur_word.get_head_vector() )
            nb_children = cur_word.count_children()
        else:
            vectors.append( self.zero_vector )


        # ветки у текущего слова
        for ichild in range(2):
            if ichild<nb_children:
                cur_word = self.stream[0]
                vectors.append( self.one1_vector ) # битовый флаг - ребенок присутствует
                vectors.append( cur_word.children[ichild].get_head_vector() )
            else:
                vectors.append( self.zero1_vector ) # битовый флаг - ребенок отсутствует
                vectors.append( self.zero_vector ) # нулевой вектор для признаков ребенка


        # lookahead(1) слово
        if len(self.stream)>1:
            vectors.append( self.one1_vector )  # битовый флаг - слово присутствует
            vectors.append( self.stream[1].get_head_vector() )
        else:
            vectors.append( self.zero1_vector )  # битовый флаг - слово отсутствует
            vectors.append( self.zero_vector )

        # tos(0)
        nb_children = 0
        if len(self.stack)>0:
            vectors.append(self.one1_vector) # битовый флаг - в стеке есть минимум 1 токен
            tos = self.stack[0]
            vectors.append( tos.get_head_vector() )
            nb_children = tos.count_children()
        else:
            vectors.append( self.zero1_vector ) # стек пуст
            vectors.append( self.zero_vector )

        # ветки у вершины стека
        for ichild in range(2):
            if ichild<nb_children:
                vectors.append( self.one1_vector ) # битовый флаг - ребенок присутствует
                vectors.append( self.stack[0].children[ichild].get_head_vector() )
            else:
                vectors.append( self.zero1_vector ) # битовый флаг - ребенок отсутствует
                vectors.append( self.zero_vector ) # нулевой вектор для признаков ребенка

        # возвращаем объединенный вектор
        return np.concatenate( vectors )

    def calc_edge_accuracy(self, predicted_edges, true_edges):
        e1 = set(predicted_edges)
        e2 = set(true_edges)
        return len( e1 & e2 )/max( len(e1), len(e2) )

    def get_final_reward(self):
        reward = 0.0

        # TODO: переделать на анализ только корня, так как ветки мы уже оценили в ходе операци
        # со стеком.

        if len(self.stream)>=1:
            root_index = self.stream[0].get_token_index()
            root_edge = (0, root_index)
            if root_edge in self.true_edges:
                return 1.0
            else:
                return -1.0


        return -1.0;


        # 1) берем первый токен в потоке
        # 2) строим список ребер в этом дереве
        # 3) сравниваем с тем списком, который должен получится в этом эпизоде
        predicted_edges = []
        if len(self.stream)>0:
            predicted_edges = self.stream[0].get_edges()
            # считаем, что первый токен в потоке - корень всего дерева, поэтому
            # в список эталонных ребер добавляем информацию о корне
            predicted_edges.append( (0, self.stream[0].get_token_index()) )

        # список ребер, который должны были получиться
        accuracy = self.calc_edge_accuracy( predicted_edges, self.true_edges)
        reward += accuracy

        return reward

# ---------------------------------------------------------------------------


class SyntaxEnv(object):

    def __init__(self, treebank):
        self._treebank = treebank
        nb_actions = 3
        self._action_space = gym.spaces.Discrete(nb_actions)

        random_episode = treebank.choose_episode()
        tmp_episode_processor = EpisodeProcessor(random_episode)
        state_size = len(tmp_episode_processor.get_state())
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,))
        self.current_episode_processor = None

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        episode = self._treebank.choose_episode()
        self.current_episode_processor = EpisodeProcessor(episode)
        observation = self.current_episode_processor.get_state()
        return observation

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        reward = self.current_episode_processor.perform_action(action)
        done = self.current_episode_processor.is_finished()
        if done:
            reward += self.current_episode_processor.get_final_reward()
        observation = self.current_episode_processor.get_state()
        info = dict()
        return (observation, reward, done, info)

    @property
    def action_space(self):
        return self._action_space
        #raise NotImplementedError

    @property
    def observation_space(self):
        return self._observation_space
        #raise NotImplementedError

    def render(self, mode):
        pass

# ------------------------------------------------------


env = SyntaxEnv(treebank)
np.random.seed(123)
#env.seed(123)
#assert len(env.action_space.shape) == 1
nb_actions = env.action_space.n
state_size = env.observation_space.shape[0]

if True:

    # Next, we build a very simple model regardless of the dueling architecture
    # if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
    # Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(state_size/2))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(state_size/4))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(state_size/8))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    #policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=10,
                   enable_dueling_network=False,
                   dueling_type='avg',
                   target_model_update=1e-4,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)

else:
    # SARSA
    # SARSA does not require a memory.
    policy = BoltzmannQPolicy()

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(state_size/2))
    model.add(Activation('relu'))
    model.add(Dense(state_size/4))
    model.add(Activation('relu'))
    model.add(Dense(state_size/8))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    sarsa = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

    sarsa.fit(env, nb_steps=500000, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    sarsa.save_weights('weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    sarsa.test(env, nb_episodes=5, visualize=False)
