

from argparse import Action
import os,json,zlib,itertools,torch,random
from collections import deque
import numpy as np
from sapai import Player, data
from sapai.battle import Battle
from sapai.compress import compress,decompress,minimal_state
from sapai.model import Linear_QNet, QTrainer

### Pets with a random component
###   Random component in the future should just be handled in an exact way
###   whereby all possible outcomes are evaluated just once. This would 
###   significantly speed up training. 
random_buy_pets = {"pet-otter"}
random_sell_pets = {"pet-beaver"}
random_pill_pets = {"pet-ant"}
random_battle_pets = {"pet-mosquito"}


class CombinatorialSearch():
    """
    CombinatorialSearch is a method to enumerate the entire possible search
    space for the current shopping phase. The search starts from an initial 
    player state provided in the arguments. Then all possible next actions 
    are taken from that player state until the Player's gold is exhausted.
    Returned is a list of all final possible final player states after the 
    shopping phase. 
    
    This algorithm can be parallelized in the sense that after round 1, there 
    will be a large number of player states. Each of these player states can 
    then be fed back into the CombinatorialSearch individually to search for
    their round 2 possibilities. Therefore, parallelization can occur across
    the possible player states for each given round. This parallelization
    should take place outside of this Class. 
    
    Parallelization within this Class itself would be a bit more difficult and
    would only be advantageous to improve the search speed when condering a 
    single team. This would be better for run-time evaluation of models, but 
    it unnecessary for larger searchers, which are the present focus. Therefore,
    this will be left until later. 
    
    A CombinatorialAgent can be built on top of this to resemble the way that a 
    human player  will make decisions. However, the CombinatorialAgent will 
    consider all possible combinations of decisions in order arrive at the 
    best possible next decisions given the available gold. 
    
    Arguments
    ---------
    verbose: bool
        If True, messages are printed during the search
    max_actions: int
        Maximum depth, equal to the number of shop actions, that can be performed
        during search space enumeration. Using max_actions of 1 would correspond
        to a greedy search algorithm in conjunction with any Agents. 
    
    """
    def __init__(self, verbose=True, max_actions=-1):
        self.verbose = verbose 
        self.max_actions = max_actions
        
        ### This stores the player lists for performing all possible actions
        self.player_list = []
        
        ### Player dict stores compressed str of all players such that if the
        ###   same player state will never be used twice
        self.player_state_dict = {}
        
        self.current_print_number = 0
        
    
    def avail_actions(self, player):
        """
        Return all possible available actions
        
        """
        action_list = [()]
        ### Include only the actions that cost gold
        action_list += self.avail_buy_pets(player)
        action_list += self.avail_buy_food(player)
        action_list += self.avail_buy_combine(player)
        action_list += self.avail_team_combine(player)
        action_list += self.avail_sell(player)
        action_list += self.avail_sell_buy(player)
        action_list += self.avail_roll(player)
        return action_list
        
    
    def avail_buy_pets(self, player):
        """
        Returns all possible pets that can be bought from the player's shop
        """
        action_list = []
        gold = player.gold
        if len(player.team) == 5:
            ### Cannot buy for full team
            return action_list
        for shop_idx,shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "pet":
                if shop_slot.cost <= gold:
                    action_list.append((player.buy_pet,shop_idx))
        return action_list
    
    
    def avail_buy_food(self, player):
        """
        Returns all possible food that can be bought from the player's shop
        """
        action_list = []
        gold = player.gold
        if len(player.team) == 0:
            return action_list
        for shop_idx,shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "food":
                if shop_slot.cost <= gold:
                    for team_idx,team_slot in enumerate(player.team):
                        if team_slot.empty:
                            continue
                        action_list.append(
                            (player.buy_food,
                             shop_idx,
                             team_idx))
        return action_list
    
    
    def avail_buy_combine(self, player):
        action_list = []
        gold = player.gold
        team_names = {}
        for team_idx,slot in enumerate(player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(team_idx)
        if len(player.team) == 0:
            return action_list
        for shop_idx,shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "pet":
                ### Can't combine if pet not already on team
                if shop_slot.item.name not in team_names:
                    continue
                
                if shop_slot.cost <= gold:
                    for team_idx in team_names[shop_slot.item.name]:
                        action_list.append(
                            (player.buy_combine,
                             shop_idx,
                             team_idx))
        return action_list
    
    
    def avail_team_combine(self, player):
        action_list = []
        if len(player.team) == 0:
            return action_list
        
        team_names = {}
        for slot_idx,slot in enumerate(player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(slot_idx)
            
        for key,value in team_names.items():
            if len(value) == 1:
                continue
            
            for idx0,idx1 in itertools.combinations(value,r=2):
                action_list.append(
                    (player.combine,
                    idx0,
                    idx1))
        
        return action_list
    
    
    def avail_sell(self, player):
        action_list = []
        if len(player.team) <= 1:
            ### Not able to sell the final friend on team
            return action_list
        for team_idx,slot in enumerate(player.team):
            if slot.empty:
                continue
            action_list.append((player.sell, team_idx))
        return action_list
    
    
    def avail_sell_buy(self, player):
        """ 
        Sell buy should only be used if the team is full. This is done so that
        the search space is not increased unnecessarily. However, a full agent
        implementation can certainly consider this action at any point in the 
        game as long as there are pets to sell and buy 
        """
        action_list = []
        gold = player.gold
        if len(player.team) != 5:
            return action_list
        team_idx_list = player.team.get_fidx()
        shop_idx_list = []
        for shop_idx,shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "pet":
                if shop_slot.cost <= gold:
                    shop_idx_list.append(shop_idx)
        
        prod = itertools.product(team_idx_list, shop_idx_list)
        for temp_team_idx,temp_shop_idx in prod:
            action_list.append((player.sell_buy, temp_team_idx, temp_shop_idx))
            
        return action_list
    
    
    def avail_team_order(self, player):
        """ Returns all possible orderings for the team """
        action_list = []
        
        team_range = np.arange(0,len(player.team))
        if len(team_range) == 0:
            return []
        
        for order in itertools.permutations(team_range, r=len(team_range)):
            action_list.append((player.reorder, order))
        
        return action_list
    
    
    def avail_roll(self, player):
        action_list = []
        ##### ASSUMPTION: If gold is not 1, do not roll for now because with 
        #####   ShopLearn, rolling has no meaning
        if player.gold != 1:
            return action_list
        if player.gold > 0:
            action_list.append((player.roll,))
        return action_list
        
    
    def search(self, player, player_state_dict=None):
        ### Initialize internal storage
        self.player = player
        self.player_list = []
        self.player_state = self.player.state
        self.current_print_number = 0
        self.print_message("start", self.player)

        ### build_player_list searches shop actions and returns player list
        ###   In addition, it builds a player_state_dict which can be used for
        ###   faster lookup of redundant player states
        if player_state_dict == None:
            self.player_state_dict = {}
        else:
            self.player_state_dict = player_state_dict
        self.player_list = self.build_player_list(self.player)
        self.print_message("player_list_done", self.player_list)
        
        ### Now consider all possible reorderings of team
        self.player_list,self.player_state_dict = self.search_reordering(
                                    self.player_list,self.player_state_dict)
        
        ### End turn for all in player list
        for temp_player in self.player_list:
            temp_player.end_turn()
        ### NOTE: After end_turn the player_state_dict has not been updated, 
        ###   therefore, the player_state_dict is no longer reliable and should
        ###   NOT be used outside of this Class. If the player_state_dict is 
        ###   required, it should be rebuilt from the player_list itself
        
        ### Also, return only the unique team list for convenience
        self.team_dict = self.get_team_dict(self.player_list)
        
        self.print_message("done", (self.player_list, self.team_dict))
        
        return self.player_list,self.team_dict
    
    
    def build_player_list(self, player, player_list=None):
        """
        Recursive function for building player list for a given turn using all
        actions during the shopping phase

        """
        if player.gold <= 0:
            ### If gold is 0, then this is exit condition for the 
            ### recursive function
            return []
        if player_list == None:
            player_list = []

        player_state = player.state
        self.print_message("size", self.player_state_dict)
        if self.max_actions > 0:
            actions_taken = len(player.action_history)
            if actions_taken >= self.max_actions:
                return []
        
        avail_actions = self.avail_actions(player)
        for temp_action in avail_actions:
            if temp_action == ():
                ### Null action
                continue
            
            #### Re-initialize Player
            temp_player = Player.from_state(player_state)
            
            #### Perform action
            action_name = str(temp_action[0].__name__).split(".")[-1]
            action = getattr(temp_player,action_name)
            action(*temp_action[1:])
            
            ### Check if this is unique player state
            temp_player.team.move_forward()     ### Move team forward so that
                                                ### team is index invariant

            ### Don't need history in order to check for redundancy of the 
            ###   shop state. This means that it does not matter how a Shop
            ###   gets to a state, just that the state is identical to others. 
            cstate = compress(temp_player,minimal=True)
            # cstate = hash(json.dumps(temp_player.state))
            if cstate not in self.player_state_dict:
                self.player_state_dict[cstate] = temp_player
            else:
                ### If player state has been seen before, then do not append
                ### to the player list.
                continue
            
            player_list.append(temp_player)
        
        full_player_list = player_list
        for player in player_list:
            ### Now, call this function recurisvely to add the next action
            temp_player_list = []
            self.build_player_list(player, temp_player_list)
            full_player_list += temp_player_list
        
        return full_player_list
    
    
    def search_reordering(self, player_list, player_state_dict):
        """
        Searches over all possible unique reorderings of the teams
        
        """
        additional_player_list = []
        for player in player_list:
            player_state = player.state
            reorder_actions = self.avail_team_order(player)
            for temp_action in reorder_actions:
                if temp_action == ():
                    ### Null action
                    continue
                
                #### Re-initialize identical Player
                temp_player = Player.from_state(player_state)
                
                #### Perform action
                action_name = str(temp_action[0].__name__).split(".")[-1]
                action = getattr(temp_player,action_name)
                action(*temp_action[1:])
                
                ### Check if this is unique player state
                temp_player.team.move_forward()     ### Move team forward so that
                                                    ### team is index invariant

                ### Don't need history in order to check for redundancy of the 
                ###   shop state. This means that it does not matter how a Shop
                ###   gets to a state, just that the state is identical to others. 
                cstate = compress(temp_player,minimal=True)
                # cstate = hash(json.dumps(temp_player.state))
                if cstate not in player_state_dict:
                    player_state_dict[cstate] = temp_player
                else:
                    ### If player state has been seen before, then do not append
                    ### to the player list.
                    continue
                
                additional_player_list.append(temp_player)
                
                ##### METHOD SHOULD BE USED THAT DOESN'T REQUIRE Player.from_state
                #####   This would save a lot of time
                # ### Move team back into place
                # order_idx = temp_action[1]
                # reorder_idx = np.argsort(order_idx).tolist()
                # action(reorder_idx)
                # ### Delete last two actions to reset player
                # del(temp_player.action_history[-1])
                # del(temp_player.action_history[-1])
        
        player_list += additional_player_list
        return player_list,player_state_dict
    
    
    def get_team_dict(self, player_list):
        """
        Returns dictionary of only the unique teams

        """
        team_dict = {}
        for player in player_list:
            team = player.team
            ### Move forward to make team index invariant
            team.move_forward()
            cteam = compress(team,minimal=True)
            ### Can just always do like this, don't need to check if it's 
            ###   already in dictionary because it can just be overwritten
            team_dict[cteam] = team
        return team_dict
    
    
    def print_message(self, message_type, info):
        if self.verbose == False:
            return 
        
        if message_type not in ["start", "size", "player_list_done", "done"]:
            raise Exception("Unrecognized message type {}".format(message_type))
        
        if message_type == "start":
            print("---------------------------------------------------------")
            print("STARTING SEARCH WITH INITIAL PLAYER: ")
            print(info)
            
            print("---------------------------------------------------------")
            print("STARTING TO BUILD PLAYER LIST")
            
        elif message_type == "size":
            temp_size = len(info)
            if temp_size < (self.current_print_number+100):
                return
            print("RUNNING MESSAGE: Current Number of Unique Players is {}"
                  .format(len(info)))
            self.current_print_number = temp_size
        
        elif message_type == "player_list_done":
            print("---------------------------------------------------------")
            print("DONE BUILDING PLAYER LIST")
            print("NUMBER OF PLAYERS IN PLAYER LIST: {}".format(len(info)))
            
            print("BEGINNING TO SEARCH FOR ALL POSSIBLE TEAM ORDERS")
            
        elif message_type == "done":
            print("---------------------------------------------------------")
            print("DONE WITH CombinatorialSearch")
            print("NUMBER OF PLAYERS IN PLAYER LIST: {}".format(len(info[0])))
            print("NUMBER OF UNIQUE TEAMS: {}".format(len(info[1])))
        

class DatabaseLookupRanker():
    """
    Will provide a rank to a given team based on its performance on a database
    of teams. 
    
    """
    def __init__(self, 
                 path="",
                ):
        self.path = path
        
        if os.path.exists(path):
            with open(path, "r") as f:
                self.database = json.loads(f)
        else:
            self.database = {}

        self.team_database = {}
        for key,value in self.database:
            self.team_database[key] = {"team": decompress(key),
                                       "wins": int(len(self.database)*value),
                                       "total": len(self.database)-1}
    
    
    def __call__(self, team):
        c = compress(team)
        if c in self.database:
            return self.database[c]
        else:
            return self.run_against_database(team)
    
    
    def run_against_database(self, team):
        #### Add team to database
        team_key = compress(team, minimal=True)
        if team_key not in self.team_database:
            self.team_database[team_key] = {"team": team, 
                                            "wins": 0,
                                            "total": 0}
        
        for key,value in self.team_database.items():
            # print(team, value["team"])
            self.t0 = team
            self.t1 = value["team"]
            
            f = Battle(team,value["team"])
            winner = f.battle()
        
            winner_key = [[team_key],[key],[]][winner]
            for temp_key in winner_key:
                self.team_database[temp_key]["wins"] += 1
            for temp_key in [team_key,key]:
                self.team_database[temp_key]["total"] += 1
                
        for key,value in self.team_database.items():
            wins = self.team_database[key]["wins"]
            total = self.team_database[key]["total"]
            self.database[key] = wins/total
        
        wins = self.team_database[team_key]["wins"]
        total = self.team_database[team_key]["total"]
        return wins/total
        

    def test_against_database(self, team):
        wins = 0
        total = 0
        for key,value in self.team_database.items():
            # print(team, value["team"])
            f = Battle(team,value["team"])
            winner = f.battle()
            if winner == 0:
                wins += 1
            total += 1
        return wins,total
    
    
class PairwiseBattles():
    """
    Parallel function using MPI for calculation Pairwise battles. 
    
    Disadvantage of current method is that not check-pointing is done in the
    calculation. This means that the results will be written as all or nothing. 
    If the calculation is interrupted before finishing, than all results will 
    be lost. This is a common issue of simple parallelization...
    
    """
    def __init__(self, output="results.pt"):
        try: 
            from mpi4py import MPI
            parallel_check = True
        except:
            parallel_check = False
        
        if parallel_check == False:
            raise Exception("MPI parallelization not available")

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.output = output
        
    
    def battle(self, obj):
        ### Prepare job-list on rank 0 
        if self.rank == 0:
            team_list = []
            if type(obj) == dict:
                team_list += list(obj.values())
            else:
                team_list += list(obj)
            
            if type(team_list[0]).__name__ != "Team":
                raise Exception("Input object is not Team Dict or Team List")
            
            print("------------------------------------")
            print("RUNNING PAIRWISE BATTLES")
            print("{:16s}: {}".format("INFO", "NUMBER"))
            print("{:16s}: {}".format("NUM RANKS", self.size))
            print("{:16s}: {}".format("INPUT TEAMS", len(obj)))
            
            ### Easier for indexing
            team_array = np.zeros((len(team_list),), dtype=object)
            team_array[:] = team_list[:]
            pair_idx = self._get_pair_idx(team_list)
            print("{:16s}: {}".format("NUMBER BATTLES", len(pair_idx)))
            ### Should I send just index and read in files on all ranks...
            ###   or should Teams be sent to ranks...
            ### Well, I don't think this function will every have >2 GB sized 
            ###   team dataset anyways...
            for temp_rank in np.arange(1,self.size):
                temp_idx = pair_idx[temp_rank::self.size]
                temp_teams = np.take(team_array, temp_idx)
                self.comm.send((temp_idx, temp_teams), temp_rank)
                
            my_idx = pair_idx[0::self.size]
            my_teams = np.take(team_array, my_idx)
            print("{:16s}: {}".format("BATTLES PER RANK", len(my_teams)))            
            
        else:
            ### Wait for info from rank 0
            my_idx,my_teams = self.comm.recv(source=0)
    
        if self.rank != 0:
            winner_list = []
            iter_idx = 0
            for t0,t1 in my_teams:
                b = Battle(t0,t1)
                temp_winner = b.battle()
                winner_list.append(temp_winner)
                iter_idx += 1
        else:
            #### This is split to remove branching code in the for loop above
            winner_list = []
            iter_idx = 0
            for t0,t1 in my_teams:
                b = Battle(t0,t1)
                temp_winner = b.battle()
                winner_list.append(temp_winner)
                iter_idx += 1
                if iter_idx % 1000 == 0:
                    print("{:16s}: {} of {}".format(
                                "FINISHED", iter_idx*self.size, len(pair_idx)))
        
        winner_list = np.array(winner_list).astype(int)
        
        ### Send results back to rank 0
        self.comm.barrier()
        
        if self.rank == 0:
            print("------------------------------------")
            print("DONE CALCULATING BATTLES")
            ### Using +1 so that the last entry in the array can be used as 
            ###   as throw-away when draws occur
            wins = np.zeros((len(team_array)+1,)).astype(int)
            total = np.zeros((len(team_array)+1,)).astype(int)
            ### Add info from rank 0
            add_totals_idx,add_totals= np.unique(my_idx[:,0],return_counts=True)
            total[add_totals_idx] += add_totals
            add_totals_idx,add_totals= np.unique(my_idx[:,1],return_counts=True)
            total[add_totals_idx] += add_totals
            ### Use for fast indexing for counting up wins
            temp_draw_idx = np.zeros((len(my_idx),))-1
            winner_idx_mask = np.hstack([my_idx, temp_draw_idx[:,None]])
            winner_idx_mask = winner_idx_mask.astype(int)
            winner_idx = winner_idx_mask[np.arange(0,len(winner_list)),
                                            winner_list]
            winner_idx,win_count = np.unique(winner_idx, return_counts=True)
            wins[winner_idx] += win_count
            
            for temp_rank in np.arange(1,self.size):
                temp_idx,temp_winner_list = self.comm.recv(source=temp_rank)
                add_totals_idx,add_totals= np.unique(temp_idx[:,0],
                                                     return_counts=True)
                total[add_totals_idx] += add_totals
                add_totals_idx,add_totals= np.unique(temp_idx[:,1],
                                                     return_counts=True)
                total[add_totals_idx] += add_totals
                ### Use for fast indexing for counting up wins
                temp_draw_idx = np.zeros((len(temp_idx),))-1
                winner_idx_mask = np.hstack([temp_idx, temp_draw_idx[:,None]])
                winner_idx_mask = winner_idx_mask.astype(int)
                winner_idx = winner_idx_mask[np.arange(0,len(temp_winner_list)),
                                             temp_winner_list]
                winner_idx,win_count = np.unique(winner_idx, return_counts=True)
                wins[winner_idx] += win_count
            
            ### Throw away last entry for ties
            wins = wins[0:-1]
            total = total[0:-1]
            frac = wins / total
            
            results = {}
            for iter_idx,temp_team in enumerate(team_list):
                temp_frac = frac[iter_idx]
                results[compress(temp_team, minimal=True)] = temp_frac
                
            print("WRITING OUTPUTS AT: {}".format(self.output))
            torch.save(results, self.output)
            
            print("------------------------------------")
            print("COMPLETED")
        else:
            self.comm.send((my_idx, winner_list), 0)
        
        ### Barrier before exiting
        self.comm.barrier()
        return 
    
    def _get_pair_idx(self, team_list):
        """
        Get the dictionary of pair_dict that have to be made for pair mode
        esxecution.
        
        """
        idx = np.triu_indices(n=len(team_list),
                              k=1,
                              m=len(team_list)) 
        return np.array([x for x in zip(idx[0], idx[1])])
        
        



class Agent:

    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001
    ACTIONS = 5 # buy pets
    ACTIONS += 5 # sell pets
    ACTIONS += 10 # combining pets
    ACTIONS += 20 # combining buying pets
    ACTIONS += 1 # roll
    ACTIONS += 20 # buying food
    ACTIONS += 7 # freeze
    ACTIONS += 7 # unfreeze
    ACTIONS += 10 # reordering

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.MAX_MEMORY) # popleft()
        self.model = Linear_QNet(41, 256, self.ACTIONS)
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.gamma)
        self.all_pets = data['pets'].values()
        self.all_foods = data['foods'].values()

    def get_state(self, player: Player):
        # get player pet names location form data
        state = [player.lives/10]
        state.append(player.gold / 50) 
        pets = [x.pet for x in player.team]
        pet_locs = [self.get_pet_index_fraction(x.name) for x in pets]

        for i in range(5):
            if i < len(pet_locs) and pets[i].name != 'pet-none':
                state.append(pet_locs[i])
                state.append(pets[i].health / 50)
                state.append(pets[i].attack / 50)
            else:
                state.append(-1)
                state.append(-1)
                state.append(-1)

        pets = [x for x in player.shop.pets]
        pet_locs = [self.get_pet_index_fraction(x.name) for x in pets]

        foods = [x.name for x in player.shop.foods]
        food_locs = [self.get_food_index_fraction(x) for x in foods]

        for i in range(5):
            if i < len(pet_locs) and pets[i].name != 'pet-none':
                state.append(pet_locs[i])
                state.append(pets[i].health / 50)
                state.append(pets[i].attack / 50)
            else:
                state.append(-1)
                state.append(-1)
                state.append(-1)
        
        for i in range(2):
            if i < len(food_locs):
                state.append(food_locs[i])
            else:
                state.append(-1)
        frozen_state = [ 1 if x.frozen else 0 for x in player.shop.shop_slots]
        pets = sum(1 for e in player.shop.shop_slots if e.slot_type == 'pet')
        while pets < 5 and len(frozen_state) < 6:
            frozen_state.insert(3, -1)
        while len(frozen_state) < 7:
            frozen_state.append(-1)
        state.extend(frozen_state)
        return np.array(state, dtype=int)
    def get_pet_index_fraction(self, pet_name):
        if pet_name == 'pet-none':
            return -1
        return next((index for (index, d) in enumerate(self.all_pets) if d["id"] == pet_name), None)/len(self.all_pets)

    def get_food_index_fraction(self, food_name):
        if food_name == 'food-none':
            return -1
        return next((index for (index, d) in enumerate(self.all_foods) if d["id"] == food_name), None)/len(self.all_foods)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0] * self.ACTIONS
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, self.ACTIONS - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # move = torch.argmax(prediction).item()
            final_move = prediction.tolist()

        return final_move


def train():
    score1 = 0
    score2 = 0
    agent1 = Agent()
    agent2 = Agent()
    player1 = Player(lives=10)
    player2 = Player(lives=10)
    while True:
        player1.start_turn()
        player2.start_turn()
        # get old state
        old_state1, action_probs1 = get_player_actions(agent1, player1)
        old_state2, action_probs2 = get_player_actions(agent2, player2)

        player1.end_turn()
        player2.end_turn()
        battle = Battle(player1.team, player2.team)
        winner = battle.battle()
        if winner == 0:
            reward = 1
            player2.lives = player2.lives - 1
        elif winner == 1: # team 2 won
            reward = -1
            player1.lives = player1.lives -1
        else:
            reward = -0.5 # draw
        # get new state
        done = player1.lives == 0 or player2.lives == 0
        if(player1.lives <=0):
            reward = -10
        elif(player2.lives <=0):
            reward = 10
        
        new_state1 = agent1.get_state(player1)
        new_state2 = agent2.get_state(player2)
        agent1.train_short_memory(old_state1, action_probs1, reward, new_state1, done)
        
        agent2.train_short_memory(old_state2, action_probs2, reward * -1, new_state2, done)

        # remember
        agent1.remember(old_state1, action_probs1, reward, new_state1, done)
        agent2.remember(old_state2, action_probs2, reward * -1, new_state2, done)


        if done:
            if(player1.lives > player2.lives):
                score1+=1
            elif(player1.lives <  player2.lives):
                score2+=1
            
            player1 = Player(lives=10)
            player2 = Player(lives=10)
            agent1.n_games += 1
            agent1.train_long_memory()
            agent1.model.save(file_name="model_1.pth")
            
            agent2.n_games += 1
            agent2.train_long_memory()
            agent2.model.save(file_name="model_2.pth")
            # print scores and game number
            print("Game: {}  Score1: {}  Score2: {}".format(agent1.n_games, score1, score2))

def get_player_actions(agent, player):
    state_old = agent.get_state(player)
    # get move
    while player.gold > 0:
        action_probs = agent.get_action(state_old)
            
        idx = np.argmax(action_probs)
        if idx < 5:
            action = "buy"
        elif idx < 10:
            action = "sell"
        elif idx < 20:
            action = "combine"
        elif idx < 40:
            action = "combine_buy"
        elif idx < 41:
            action = "roll"
        elif idx < 61:
            action = "buy_food"
        elif idx < 68:
            action = "freeze"
        elif idx < 75:
            action = "unfreeze"
        elif idx < 85:
            action = "reorder"
        if tryAction(action, player, action_probs):
            if action == "roll":
                reward = 0.01
            else:
                reward = 0.1
            state_new = agent.get_state(player)
            agent.train_short_memory(state_old, action_probs, reward, state_new, False)
            agent.remember(state_old, action_probs, reward, state_new, False)
        else:
            state_new = state_old
            agent.train_short_memory(state_old, action_probs, -1, state_new, False)
            agent.remember(state_old, action_probs, -1, state_new, False)
        state_old = state_new

    return state_new, action_probs


def permutations(arr):
    """permutations of 5 choose 2"""
    if len(arr) == 2:
        return [arr, [arr[1], arr[0]]]
    else:
        result = []
        for i in range(len(arr)):
            for j in permutations(arr[:i] + arr[i+1:]):
                result.append([arr[i]] + j)
        return result

def tryAction(action, player: Player, action_probs):
    idx = np.argmax(action_probs)
    try:
        if action == "buy":
            player.buy_pet(player.shop.pets[idx])
        elif action == "sell":
            slot = player.team[idx - 5]
            if slot.pet.name == "pet-none":
                raise Exception("Pet is none") 
            player.sell(int(idx - 5))
        elif action == "combine":
            comb = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            player.combine(comb[idx -10][0], comb[idx-10][1])
        elif action == "combine_buy":
            """get permutations of 5 choose 2"""
            perm = [each_permutation for each_permutation in itertools.permutations([0, 1, 2, 3, 4], 2)]
            player.buy_combine(perm[idx - 20][0], perm[idx - 20][1])
        elif action == "roll":
            player.roll()
        elif action == "buy_food":
            idx = idx - 41
            food_idx = idx // 5
            food = player.shop.foods[food_idx]
            pet_idx = idx % 5
            player.buy_food(food, int(pet_idx))
        elif action == "freeze":
            idx = idx - 61
            player.freeze(int(idx))
        elif action == "unfreeze":
            idx = idx - 68
            player.shop.unfreeze(int(idx))
        elif action == "reorder":
            """get permutations of 5 choose 2"""
            idx = idx-75
            comb = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            idxs = [0,1,2,3,4]
            pair = comb[idx]
            tmp = idxs[pair[0]]
            idxs[pair[0]] = idxs[pair[1]]
            idxs[pair[1]] = tmp

            player.reorder(idxs)
        return True
    except Exception as inst:
        return False

if __name__ == '__main__':
    train()

