from pypokerengine.api.emulator import Emulator
import players
import dqn 
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def print_events(events):
   for e in events:
      print(e)
      print()
   print()
   return


def setup():
   # create tensorboard log
   writer = SummaryWriter()

   # setup variables
   player_num = 5
   rounds = 1
   small_blind_amount = 15
   ante_amount = 10
   stack = 100

   # agent parameters
   episodes = 5000
   memory_size = 100000
   update_freq = 50
   batch_size = 32
   eps_start = 1.0
   eps_end = 0.1
   discount = 0.995
   target_update_freq = update_freq * 2

   # save model every 1000 episodes
   save_freq = 1000
   save_dir = 'models/model_3_weights'

   # check save directory
   if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

   # create emulator environment
   emulator = Emulator()
   emulator.set_game_rule(
                        player_num=player_num, 
                        max_round=rounds, 
                        small_blind_amount=small_blind_amount, 
                        ante_amount=ante_amount
   )

   # create players
   player_info = {
      '1': {'name': 'CallPlayer', 'stack': stack},
      '2': {'name': 'FoldPlayer', 'stack': stack},
      '3': {'name': 'HeuristicPlayer', 'stack': stack},
      '4': {'name': 'RandomPlayer', 'stack': stack},
      '5': {'name': 'DQN', 'stack': stack}
   }

   # create agent
   agent = dqn.DQNPlayer_Eric('5', player_num, stack, memory_size, update_freq, batch_size, eps_start, eps_end,
                              discount, target_update_freq)

   # setup poker emulator
   emulator.register_player('1', players.CallPlayer())
   emulator.register_player('2', players.FoldPlayer())
   emulator.register_player('3', players.HeuristicPlayer())
   emulator.register_player('4', players.RandomPlayer())
   emulator.register_player('5', agent)

   for i in range(episodes):

      # start game
      inital_state = emulator.generate_initial_game_state(player_info)
      game_state, events = emulator.start_new_round(inital_state)

      # play round of poker
      while not emulator._is_last_round(game_state, emulator.game_rule):
         game_state, events = emulator.run_until_round_finish(game_state)
         
         final_stack = events[-1]['players'][4]['stack']
         final_event = events[-2]
         # pot = final_event['round_state']['pot']['main']['amount']
         round_state = final_event['round_state']

         # normalize reward based on stack size
         reward = (final_stack - stack) / stack

         # update network
         loss = agent.update(i, reward, round_state)

         if final_event['winners'][0]['uuid'] == '5':
            print('Episode = {}, Won = Y, Stats = {}, Loss = {}'.format(i + 1, events[-1]['players'][4], loss))
         else:
            print('Episode = {}, Won = N, Stats = {}, Loss = {}'.format(i + 1, events[-1]['players'][4], loss))

         if loss is not None:
            writer.add_scalar('Loss/Episode', loss, i)

      # save model
      if i % save_freq == 0:
         path = os.path.join(save_dir, 'eric/model_' + str(i) + '.pth')
         torch.save(agent.model_predict.state_dict(), path)

   # save final model iteration
   path = os.path.join(save_dir, 'eric/model_' + str(episodes - 1) + '.pth')
   torch.save(agent.model_predict.state_dict(), path)

   writer.flush()
   writer.close()


if __name__ == '__main__':
   setup()
