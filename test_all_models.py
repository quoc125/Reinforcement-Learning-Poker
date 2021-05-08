from pypokerengine.api.game import setup_config, start_poker
import dqn
import players 

def test():
   # game parameters
   player_num = 5
   max_round = 1
   small_blind_amount = 15
   ante_amount = 10
   initial_stack = 100

   # agent parameters needed to intialize agent
   episodes = 500 # number of times to test agent only parameter needed

   memory_size = 100000
   update_freq = 50
   batch_size = 32
   eps_start = 1.0
   eps_end = 0.1
   discount = 0.995
   target_update_freq = update_freq * 2

   # setup dqn agent
   agent1 = dqn.DQNPlayer_Eric('3', player_num, initial_stack, memory_size, update_freq, batch_size, eps_start,
                               eps_end, discount, target_update_freq, 'models/eric/model_4000.pth')

   agent2 = dqn.DQNPlayer_Quoc(player_num, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                               target_update_freq, 'models/quoc/model_3000.pth')

   agent3 = dqn.DQNPlayer_Saeed(player_num, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                                target_update_freq, 'models/saeed/model_1000.pth')

   # setup poker environment
   config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)

   #config.register_player(name='1', algorithm=players.CallPlayer())
   #config.register_player(name='2', algorithm=players.FoldPlayer())
   #config.register_player(name='3', algorithm=players.HeuristicPlayer())
   #config.register_player(name='4', algorithm=players.RandomPlayer())
   config.register_player(name='1', algorithm=players.HeuristicPlayer())
   config.register_player(name='2', algorithm=players.HeuristicPlayer())
   config.register_player(name='3', algorithm=agent1)
   config.register_player(name='4', algorithm=agent2)
   config.register_player(name='5', algorithm=agent3)

   games_won_eric = 0
   games_won_quoc = 0
   games_won_saeed = 0

   stack_avg_eric = 0
   stack_avg_quoc = 0
   stack_avg_saeed = 0

   # test poker agent
   for i in range(episodes):
      game_result = start_poker(config, verbose=0)
      
      stack = 0
      winner = None

      stack_eric = game_result['players'][2]['stack']
      stack_quoc = game_result['players'][3]['stack']
      stack_saeed = game_result['players'][4]['stack']

      stack_diff_eric = stack_eric - initial_stack
      stack_diff_quoc = stack_quoc - initial_stack
      stack_diff_saeed = stack_saeed - initial_stack

      stack_avg_eric += stack_diff_eric
      stack_avg_quoc += stack_diff_quoc
      stack_avg_saeed += stack_diff_saeed

      for player in game_result['players']:
         if player['stack'] > stack:
            winner = player['name']
            stack = player['stack']

      if winner == '3':
         games_won_eric += 1
      elif winner == '4':
         games_won_quoc += 1
      elif winner == '5':
         games_won_saeed += 1

      print('Round = {},\tWinner = {}'.format(i + 1, winner))
      print('\t[Eric]: Stack Diff. = {}'.format(stack_diff_eric))
      print('\t[Quoc]: Stack Diff. = {}'.format(stack_diff_quoc))
      print('\t[Saeed]: Stack Diff. = {}'.format(stack_diff_saeed))
   print('Total Episodes = {}'.format(episodes))
   print('\t[Eric]: Games Won = {},\tWinning Ratio = {},\tStack Avg. = {}'.format(games_won_eric,
                                                                                  games_won_eric / episodes,
                                                                                  stack_avg_eric / episodes))
   print('\t[Quoc]: Games Won = {},\tWinning Ratio = {},\tStack Avg. = {}'.format(games_won_quoc,
                                                                                  games_won_quoc / episodes,
                                                                                  stack_avg_quoc / episodes))
   print('\t[Saeed]: Games Won = {},\tWinning Ratio = {},\tStack Avg. = {}'.format(games_won_saeed,
                                                                                   games_won_saeed / episodes,
                                                                                   stack_avg_saeed / episodes))


if __name__ == '__main__':
   test()