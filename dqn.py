from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand

from network import LinearNetwork_Saeed, LinearNetwork_Eric, LinearNetwork_Quoc
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from collections import deque


class DQNPlayer_Saeed(BasePokerPlayer):

   def __init__(self, player_num, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                target_update_freq, model_path=None):

      self.model_predict = LinearNetwork_Saeed().cuda()
      self.model_target = LinearNetwork_Saeed().cuda()

      # q learning parameters
      self.player_num = player_num
      self.memory_size = memory_size
      self.update_freq = update_freq
      self.batch_size = batch_size
      self.replay_memory = deque(maxlen=self.memory_size)
      self.target_update_freq = target_update_freq
      self.eps_start = eps_start
      self.eps_end = eps_end
      self.discount = discount
      self.hand = None
      self.current_features = None

      # network parameters
      self.lr = 0.001

      self.model_path = model_path
      if self.model_path is not None:
         checkpoint = torch.load(self.model_path)
         self.model_predict.load_state_dict(checkpoint)

      self.model_target.load_state_dict(self.model_predict.state_dict())
      self.optimizer = Adam(self.model_predict.parameters(), lr=self.lr)
      self.loss = MSELoss()


   # get value of current hand using monte carlo simulation
   def get_estimate(self, hole_card, community_card):
      return estimate_hole_card_win_rate(
         nb_simulation=1000,
         nb_player=self.player_num,
         hole_card=gen_cards(hole_card),
         community_card=gen_cards(community_card)
      )


   def declare_action(self, valid_actions, hole_card, round_state):
      # get current community cards
      community = round_state['community_card']

      # one hot encode hole cards and community cards
      hole_encoder = self.one_hot_encode(hole_card, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)

      # get estimate of how good hand is using monte carlo simulation
      estimate = self.get_estimate(hole_card, community)

      # store hand
      if self.hand is None:
         self.hand = hole_card

      # features used for model are hole cards, community cards, and estimation
      features = self.create_features(hole_encoder, community_encoder, estimate)
      if self.current_features == None:
         self.current_features = features

      if self.model_path is None:
         # take random action
         if rand.random() < self.eps_start:
            best_action = rand.randint(0, 3)

         # use model to get action
         else:
            # get predicted action from neural network
            actions = self.model_predict.forward(features.cuda())
            best_action = actions.argmax(1)[0]

      else:
         actions = self.model_predict.forward(features.cuda())
         best_action = actions.argmax(1)[0]

      # print(best_action)

      # get action from index of best probability
      # fold
      if best_action == 0:
         action, amount = valid_actions[0]['action'], valid_actions[0]['amount']

      # call
      elif best_action == 1:
         action, amount = valid_actions[1]['action'], valid_actions[1]['amount']

      # raise min
      elif best_action == 2:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']

      # raise max
      elif best_action == 3:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']

      return action, amount


   def update(self, episode_num, community, reward):
      # update epsilon
      if self.eps_start > self.eps_end:
         self.eps_start *= self.discount

      # get state encoding
      hand_encoder = self.one_hot_encode(self.hand, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)
      estimate = self.get_estimate(self.hand, community)

      # store in memory
      features = self.create_features(hand_encoder, community_encoder, estimate)
      self.replay_memory.append([self.current_features, features])

      # reset hand and features
      self.hand = None
      self.current_features = None

      loss = None

      # update model
      if episode_num % self.update_freq == 0:
         # not enough training samples
         if len(self.replay_memory) < self.batch_size:
            return

         # get batch from replay_memory
         batch = rand.sample(self.replay_memory, self.batch_size)

         # get predictions for start states
         start_states = [transition[0] for transition in batch]
         start_q = [self.model_predict.forward(state.cuda()) for state in start_states]

         # get predictions for current end state
         end_states = [transition[1] for transition in batch]
         end_q = [self.model_target.forward(state.cuda()) for state in end_states]

         # get inputs and labels for training
         x, y = [], []

         for i in range(self.batch_size):
            x.append(start_q[i][0])

            # get max of end q values
            new_q = max(end_q[i][0])
            index = int(end_q[i].argmax(1)[0])

            ### alters q value too much ###
            #new_q = (new_q * self.discount) + reward

            # replace start q value
            update_q = start_q[i].clone()
            update_q[0][index] = new_q

            y.append(update_q[0])

         # convert lists to tensors
         ### might be a better way ###
         x = torch.cat(x, dim=0)
         y = torch.cat(y, dim=0)

         # train the model
         loss = self.loss(x, y)
         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         # update target model weights
         if episode_num % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model_predict.state_dict())


      return loss


   def create_features(self, hole_encoder, community_encoder, estimate):
      # combine features into list and convert to tensor
      features = []
      for encoding in hole_encoder:
         features.extend(encoding)

      for encoding in community_encoder:
         features.extend(encoding)

      features.extend([estimate])
      features = torch.FloatTensor([features])

      return features


   def one_hot_encode(self, cards, is_community):
      suits = {'H': 0, 'C': 1, 'D': 2, 'S': 3}
      values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

      encoding = []
      for index, card in enumerate(cards):

         # create empty lists for one hot encoding
         s = [0 for i in range(4)]
         v = [0 for i in range(13)]

         # use index to one hot encode
         one_hot_suit = suits[card[0]]
         one_hot_val = values[card[1]]
         s[one_hot_suit] = 1
         v[one_hot_val] = 1

         s.extend(v)
         encoding.append(s)

      if is_community:
         # if community card size is not 5, add empty arrays to encoding
         while len(encoding) < 5:
            encoding.append([0 for i in range(17)])

      return encoding



   def receive_game_start_message(self, game_info):
      pass

   def receive_round_start_message(self, round_count, hole_card, seats):
      pass

   def receive_street_start_message(self, street, round_state):
      pass

   def receive_game_update_message(self, action, round_state):
      pass

   def receive_round_result_message(self, winners, hand_info, round_state):
      pass


class DQNPlayer_Eric(BasePokerPlayer):

   def __init__(self, name, player_num, initial_stack, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                target_update_freq, model_path=None):

      self.model_predict = LinearNetwork_Eric().cuda()
      self.model_target = LinearNetwork_Eric().cuda()

      # q learning parameters
      self.name = name
      self.player_num = player_num
      self.total_stack = player_num * initial_stack
      self.memory_size = memory_size
      self.update_freq = update_freq
      self.batch_size = batch_size
      self.replay_memory = deque(maxlen=self.memory_size)
      self.target_update_freq = target_update_freq
      self.eps_start = eps_start
      self.eps_end = eps_end
      self.discount = discount
      self.hand = None
      self.current_features = None

      # network parameters
      self.lr = 0.001

      self.model_path = model_path
      if self.model_path is not None:
         checkpoint = torch.load(self.model_path)
         self.model_predict.load_state_dict(checkpoint)

      self.model_target.load_state_dict(self.model_predict.state_dict())
      self.optimizer = Adam(self.model_predict.parameters(), lr=self.lr)
      self.loss = MSELoss()


   # get value of current hand using monte carlo simulation
   def get_estimate(self, hole_card, community_card):
      return estimate_hole_card_win_rate(
         nb_simulation=1000,
         nb_player=self.player_num,
         hole_card=gen_cards(hole_card),
         community_card=gen_cards(community_card)
      )


   def declare_action(self, valid_actions, hole_card, round_state):
      # get current community cards
      community = round_state['community_card']

      # one hot encode hole cards and community cards
      hole_encoder = self.one_hot_encode(hole_card, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)

      # get estimate of how good hand is using monte carlo simulation
      estimate = self.get_estimate(hole_card, community)

      # get the small blind and big blind
      small_blind = round_state['small_blind_pos']
      big_blind = round_state['big_blind_pos']

      # get current state of the street
      street_dict = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
      street = street_dict[round_state['street']]

      # get current stack of the agent
      player_stack = 0
      for player in round_state['seats']:
         if player['name'] == self.name:
            player_stack = player['stack']
            break

      # get current ratio of the player stack and the total stack
      player_stack = player_stack / self.total_stack

      # get current ratio pot amount and the total stack
      pot = round_state['pot']['main']['amount'] / self.total_stack

      # store hand
      if self.hand is None:
         self.hand = hole_card

      # features used for model are hole cards, community cards, and estimation
      features = self.create_features(hole_encoder, community_encoder, estimate, small_blind, big_blind, street,
                                      player_stack, pot)
      if self.current_features == None:
         self.current_features = features

      if self.model_path is None:
         # take random action
         if rand.random() < self.eps_start:
            best_action = rand.randint(0, 3)

         # use model to get action
         else:
            # get predicted action from neural network
            actions = self.model_predict.forward(features.cuda())
            best_action = actions.argmax(1)[0]

      else:
         actions = self.model_predict.forward(features.cuda())
         best_action = actions.argmax(1)[0]

      # print(best_action)

      # get action from index of best probability
      # fold
      if best_action == 0:
         action, amount = valid_actions[0]['action'], valid_actions[0]['amount']

      # call
      elif best_action == 1:
         action, amount = valid_actions[1]['action'], valid_actions[1]['amount']

      # raise min
      elif best_action == 2:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']

      # raise max
      elif best_action == 3:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']

      return action, amount


   def update(self, episode_num, reward, round_state):
      # update epsilon
      if self.eps_start > self.eps_end:
         self.eps_start *= self.discount

      # get current community cards
      community = round_state['community_card']

      # get state encoding
      hand_encoder = self.one_hot_encode(self.hand, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)
      estimate = self.get_estimate(self.hand, community)

      # get the small blind and big blind
      small_blind = round_state['small_blind_pos']
      big_blind = round_state['big_blind_pos']

      # get current state of the street
      street_dict = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
      street = street_dict[round_state['street']]

      # get current stack of the agent
      player_stack = 0
      for player in round_state['seats']:
         if player['name'] == self.name:
            player_stack = player['stack']
            break

      # get current ratio of the player stack and the total stack
      player_stack = player_stack / self.total_stack

      # get current ratio pot amount and the total stack
      pot = round_state['pot']['main']['amount'] / self.total_stack

      # store in memory
      features = self.create_features(hand_encoder, community_encoder, estimate, small_blind, big_blind, street,
                                      player_stack, pot)
      self.replay_memory.append([self.current_features, features])

      # reset hand and features
      self.hand = None
      self.current_features = None

      loss = None

      # update model
      if episode_num % self.update_freq == 0:
         # not enough training samples
         if len(self.replay_memory) < self.batch_size:
            return

         # get batch from replay_memory
         batch = rand.sample(self.replay_memory, self.batch_size)

         # get predictions for start states
         start_states = [transition[0] for transition in batch]
         start_q = [self.model_predict.forward(state.cuda()) for state in start_states]

         # get predictions for current end state
         end_states = [transition[1] for transition in batch]
         end_q = [self.model_target.forward(state.cuda()) for state in end_states]

         # get inputs and labels for training
         x, y = [], []

         for i in range(self.batch_size):
            x.append(start_q[i][0])

            # get max of end q values
            new_q = max(end_q[i][0])
            index = int(end_q[i].argmax(1)[0])

            ### alters q value too much ###
            new_q = (new_q * self.discount) + reward

            # replace start q value
            update_q = start_q[i].clone()
            update_q[0][index] = new_q

            y.append(update_q[0])

         # convert lists to tensors
         ### might be a better way ###
         x = torch.cat(x, dim=0)
         y = torch.cat(y, dim=0)

         # train the model
         loss = self.loss(x, y)
         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         # update target model weights
         if episode_num % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model_predict.state_dict())


      return loss


   def create_features(self, hole_encoder, community_encoder, estimate, small_blind, big_blind, street, player_stack,
                       pot):
      # combine features into list and convert to tensor
      features = []
      for encoding in hole_encoder:
         features.extend(encoding)

      for encoding in community_encoder:
         features.extend(encoding)

      features.extend([estimate, small_blind, big_blind, street, player_stack, pot])
      features = torch.FloatTensor([features])

      return features


   def one_hot_encode(self, cards, is_community):
      suits = {'H': 0, 'C': 1, 'D': 2, 'S': 3}
      values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

      encoding = []
      for index, card in enumerate(cards):

         # create empty lists for one hot encoding
         s = [0 for i in range(4)]
         v = [0 for i in range(13)]

         # use index to one hot encode
         one_hot_suit = suits[card[0]]
         one_hot_val = values[card[1]]
         s[one_hot_suit] = 1
         v[one_hot_val] = 1

         s.extend(v)
         encoding.append(s)

      if is_community:
         # if community card size is not 5, add empty arrays to encoding
         while len(encoding) < 5:
            encoding.append([0 for i in range(17)])

      return encoding



   def receive_game_start_message(self, game_info):
      pass

   def receive_round_start_message(self, round_count, hole_card, seats):
      pass

   def receive_street_start_message(self, street, round_state):
      pass

   def receive_game_update_message(self, action, round_state):
      pass

   def receive_round_result_message(self, winners, hand_info, round_state):
      pass


class DQNPlayer_Quoc(BasePokerPlayer):

   def __init__(self, player_num, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                target_update_freq, model_path=None):

      self.model_predict = LinearNetwork_Quoc().cuda()
      self.model_target = LinearNetwork_Quoc().cuda()

      # q learning parameters
      self.player_num = player_num
      self.memory_size = memory_size
      self.update_freq = update_freq
      self.batch_size = batch_size
      self.replay_memory = deque(maxlen=self.memory_size)
      self.target_update_freq = target_update_freq
      self.eps_start = eps_start
      self.eps_end = eps_end
      self.discount = discount
      self.hand = None
      self.current_features = None

      # network parameters
      self.lr = 0.001

      self.model_path = model_path
      if self.model_path is not None:
         checkpoint = torch.load(self.model_path)
         self.model_predict.load_state_dict(checkpoint)

      self.model_target.load_state_dict(self.model_predict.state_dict())
      self.optimizer = Adam(self.model_predict.parameters(), lr=self.lr)
      self.loss = MSELoss()

   # get value of current hand using monte carlo simulation
   def get_estimate(self, hole_card, community_card):
      return estimate_hole_card_win_rate(
         nb_simulation=1000,
         nb_player=self.player_num,
         hole_card=gen_cards(hole_card),
         community_card=gen_cards(community_card)
      )

   def declare_action(self, valid_actions, hole_card, round_state):

      # get current community cards
      community = round_state['community_card']
      self.small_blind_pos = round_state['small_blind_pos']
      self.big_blind_pos = round_state['big_blind_pos']

      # one hot encode hole cards and community cards
      hole_encoder = self.one_hot_encode(hole_card, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)

      # get estimate of how good hand is using monte carlo simulation
      estimate = self.get_estimate(hole_card, community)

      # store hand
      if self.hand is None:
         self.hand = hole_card

      # features used for model are hole cards, community cards, and estimation
      features = self.create_features(hole_encoder, community_encoder, self.small_blind_pos, self.big_blind_pos,
                                      estimate)
      if self.current_features == None:
         self.current_features = features

      if self.model_path is None:
         # take random action
         if rand.random() < self.eps_start:
            best_action = rand.randint(0, 3)

         # use model to get action
         else:
            # get predicted action from neural network
            actions = self.model_predict.forward(features.cuda())
            best_action = actions.argmax(1)[0]

      else:
         actions = self.model_predict.forward(features.cuda())
         best_action = actions.argmax(1)[0]

      # print(best_action)

      # get action from index of best probability
      # fold
      if best_action == 0:
         action, amount = valid_actions[0]['action'], valid_actions[0]['amount']

      # call
      elif best_action == 1:
         action, amount = valid_actions[1]['action'], valid_actions[1]['amount']

      # raise min
      elif best_action == 2:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']

      # raise max
      elif best_action == 3:
         action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']

      return action, amount

   def update(self, episode_num, community, reward):
      # update epsilon
      if self.eps_start > self.eps_end:
         self.eps_start *= self.discount

      # get state encoding
      hand_encoder = self.one_hot_encode(self.hand, is_community=False)
      community_encoder = self.one_hot_encode(community, is_community=True)
      estimate = self.get_estimate(self.hand, community)

      # store in memory
      features = self.create_features(hand_encoder, community_encoder, self.small_blind_pos, self.big_blind_pos,
                                      estimate)
      self.replay_memory.append([self.current_features, features])

      # reset hand and features
      self.hand = None
      self.current_features = None

      loss = None

      # update model
      if episode_num % self.update_freq == 0:
         # not enough training samples
         if len(self.replay_memory) < self.batch_size:
            return

         # get batch from replay_memory
         batch = rand.sample(self.replay_memory, self.batch_size)

         # get predictions for start states
         start_states = [transition[0] for transition in batch]
         start_q = [self.model_predict.forward(state.cuda()) for state in start_states]

         # get predictions for current end state
         end_states = [transition[1] for transition in batch]
         end_q = [self.model_target.forward(state.cuda()) for state in end_states]

         # get inputs and labels for training
         x, y = [], []

         for i in range(self.batch_size):
            x.append(start_q[i][0])

            # get max of end q values
            new_q = max(end_q[i][0])
            index = int(end_q[i].argmax(1)[0])

            ### alters q value too much ###
            new_q = (new_q * self.discount) + reward

            # replace start q value
            update_q = start_q[i].clone()
            update_q[0][index] = new_q

            y.append(update_q[0])

         # convert lists to tensors
         ### might be a better way ###
         x = torch.cat(x, dim=0)
         y = torch.cat(y, dim=0)

         # train the model
         loss = self.loss(x, y)
         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         # update target model weights
         if episode_num % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model_predict.state_dict())

      return loss

   def create_features(self, hole_encoder, community_encoder, small_blind, big_blind, estimate):
      # combine features into list and convert to tensor
      features = []
      for encoding in hole_encoder:
         features.extend(encoding)

      for encoding in community_encoder:
         features.extend(encoding)

      features.extend([small_blind, big_blind, estimate])
      features = torch.FloatTensor([features])

      return features

   def one_hot_encode(self, cards, is_community):
      suits = {'H': 0, 'C': 1, 'D': 2, 'S': 3}
      values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                'A': 12}

      encoding = []
      for index, card in enumerate(cards):
         # create empty lists for one hot encoding
         s = [0 for i in range(4)]
         v = [0 for i in range(13)]

         # use index to one hot encode
         one_hot_suit = suits[card[0]]
         one_hot_val = values[card[1]]
         s[one_hot_suit] = 1
         v[one_hot_val] = 1

         s.extend(v)
         encoding.append(s)

      if is_community:
         # if community card size is not 5, add empty arrays to encoding
         while len(encoding) < 5:
            encoding.append([0 for i in range(17)])

      return encoding

   def receive_game_start_message(self, game_info):
      pass

   def receive_round_start_message(self, round_count, hole_card, seats):
      pass

   def receive_street_start_message(self, street, round_state):
      pass

   def receive_game_update_message(self, action, round_state):
      pass

   def receive_round_result_message(self, winners, hand_info, round_state):
      pass
