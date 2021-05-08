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
   episodes = 100 # number of times to test agent only parameter needed

   memory_size = 100000
   update_freq = 50
   batch_size = 32
   eps_start = 1.0
   eps_end = 0.1
   discount = 0.995
   target_update_freq = update_freq * 2

   # setup dqn agent
   agent = dqn.DQNPlayer_Quoc(player_num, memory_size, update_freq, batch_size, eps_start, eps_end, discount,
                              target_update_freq, 'models/quoc/model_3000.pth')
   
   # setup poker environment   
   config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)

   config.register_player(name='1', algorithm=players.CallPlayer())
   config.register_player(name='2', algorithm=players.FoldPlayer())
   config.register_player(name='3', algorithm=players.HeuristicPlayer())
   config.register_player(name='4', algorithm=players.RandomPlayer())
   config.register_player(name='5', algorithm=agent)

   # test poker agent
   games_won = 0
   stack_average = 0
   for i in range(episodes):
      game_result = start_poker(config, verbose=0)
      
      stack = 0
      winner = None
      my_stack = game_result['players'][player_num-1]['stack']
      stack_diff = my_stack - initial_stack

      stack_average += stack_diff

      for player in game_result['players']:
         if player['stack'] > stack:
            winner = player['name']
            stack = player['stack']

      if winner == '5':
         games_won += 1

      print('round = ' + str(i), end = '\t')
      print('winner = ' + winner, end = '\t')
      print('my stack difference = ' + str(stack_diff))
   print('games won = ' + str(games_won), end = '\t')
   print('total games = ' + str(episodes), end = '\t')
   print('winning = ' + str(games_won / episodes), end = '\t')
   print('stack average change = ' + str(stack_average / episodes))


if __name__ == '__main__':
   test()