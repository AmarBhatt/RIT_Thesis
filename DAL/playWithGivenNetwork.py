from numpy import *
import numpy as np
from sys import exit
from pylab import *
from os import mkdir
from os.path import exists
from random import randint
from player_agent import PlayerAgent
from samples_manager import SamplesManager
from run_ale import run_ale
from common_constants import actions_map
from game_settings import AstrixSettings, SpaceInvadersSettings, FreewaySettings, SeaquestSettings, BreakoutSettings
from memory_database import MemoryDatabase
from scipy.misc import imresize
from deepmind_network import DeepMindNetwork
import generalUtils
import theano
import theano.tensor as T
from scipy import stats
import argparse

class DeepMindAgent(PlayerAgent):
"""
A player agent that acts randomly in a game, and gathers samples.
These samples are later used to detect the game background, and
also detect object classes.

Instance Variables:
	- num_samples Number of samples to collect
	- act_contin_count Number of frames we should repeat a
		randomly selected action
	- samples_count Number of samples we have collected
		so far
	- reset_count_down sometimes we need to send reset
		action for a number of frames
	- curr_action The action taken in the previous step
	- curr_action_count Number of times we have taken this action
	- samples_manager Instance of SampleManager class, responsible
		for gathering samples
	- rand_run_only When true, we will just do a random-run, without
		actually gathering any samples
"""
	def __init__(self, game_settings, which_network, plot_histogram, non_linearity, play_for_frames,
		action_continuity_count, epsilon, working_directory = ".",
		rand_run_only = False, uniform_over_game_space = False,
		greedy_action_selection = False, history_size = 4, batch_size = 32):
		PlayerAgent.__init__(self, game_settings, working_directory)
		self.color_map = generalUtils.generateColorMap()
		#TODO: Update this, to properly support the new GameSettings framework
		self.act_contin_count = action_continuity_count
		self.curr_action = None
		self.curr_action_count = 0
		self.restart_delay = 0
		self.initial_delay = 100
		self.episode_status = 'uninitilized'
		self.episode_counter = 0
		self.episode_reward =0
		self.curr_state = []
		self.frame_count = 0
		self.database = MemoryDatabase(history_size = history_size)
		self.uniform_over_game_space = uniform_over_game_space
		self.greedy_action_selection = greedy_action_selection
		self.save_next_frame = False
		self.prev_state = None
		self.prev_action = None
		self.influence = np.zeros((256, 83, 83))
		self.history_size = history_size
		self.batch_size = batch_size
		self.batch = np.zeros((self.batch_size, self.history_size, 83, 83))
		self.plot_histogram = plot_histogram
		x = T.tensor4('x')
		y = T.ivector('y')
		rng = np.random.RandomState(23455)
		net = DeepMindNetwork.readFromFile(
			x = x,
			num_actions = len(game_settings.possible_actions),
			batch_size = self.batch_size,
			rng = rng,
			fileName = which_network,
			history_size = self.history_size,
			non_linearity = non_linearity)

		self.get_action = theano.function(
			[x],
			net.layer3.y_pred[0])

		self.get_probs = theano.function(
			[x],
			net.layer3.p_y_given_x[0])
		self.epsilon = epsilon

		#self.background = np.load('data/background0.npz')['background']
		self.a = []
		i = T.iscalar('i')
		self.get_influence = theano.function([x, i], T.grad(net.layer3.p_y_given_x[0][i], x),
			allow_input_downcast=True)
		#self.get_influence = theano.function([x, i], T.grad(net.layer3.p_y_given_x[0][i], x), allow_input_downcast=True) self.get_influence = theano.function([x, i], T.grad(net.layer3. p_y_given_x[0][i], x), allow_input_downcast=True)
		self.get_outputs = theano.function([x, i], net.layer1.output[0][i], allow_input_downcast=True)

	def export_image(self, screen_matrix, filename):
		"Saves the given screen matrix as a png file"
		try:
			from PIL import Image
		except ImportError:
			exit("Unable to import PIL. Python Image Library is required for" +
			" exporting screen matrix to PNG files")
		plot_height, plot_width = screen_matrix.shape[0], screen_matrix.shape[1]
		rgb_array = zeros((plot_height, plot_width , 3), uint8)
		counter = 0
		for i in range(plot_height):
			for j in range(plot_width):
				rgb_array[i,j,0] = screen_matrix[i, j]
				rgb_array[i,j,1] = screen_matrix[i, j]
				rgb_array[i,j,2] = screen_matrix[i, j]
		pilImage = Image.fromarray(rgb_array, 'RGB')
		pilImage.save(filename)

	def export_color_image(self, screen_matrix, filename):
		"Saves the given screen matrix as a png file"
		try:
			from PIL import Image
		except ImportError:
			exit("Unable to import PIL. Python Image Library is required for" +
			" exporting screen matrix to PNG files")
		pilImage = Image.fromarray(screen_matrix, 'RGB')
		pilImage.save(filename)

	def scale_array(self, rawpoints):

		high = 256.0
		low = 0.0
		mins = np.min(rawpoints)
		maxs = np.max(rawpoints)
		rng = maxs - mins
		scaled_points = high - (((high - low) * (maxs - rawpoints)) / rng)
		return scaled_points

	def add_to_batch(self, new_frame):
		# Shift previous frames
		self.batch[0, :self.history_size - 1] = self.batch[0, 1:]
		# Add new frame
		self.batch[0, self.history_size - 1] = new_frame

	def agent_step(self, screen_matrix, console_ram, reward = None):
	"""
		The main method. Given a 2D array of the color indecies on the
		screen (and potentially the reward recieved), this method
		will decides the next action based on the learning algorithm.
		Here, we are using random actions, and we save each new
	"""
		# See if we ar in the inital-delay period.
		if self.initial_delay > 0:
			# We do nothing, until the game is ready to be restarted.
			self.initial_delay -= 1
			print "Initial delay:", self.initial_delay
			return actions_map['player_a_noop']
		# At the very begining, we have to restart the game
		if self.episode_status == "uninitilized":
			if self.game_settings.first_action is not None:
				# Perform the very first action next (this is hard-coded)
				self.episode_status = "first_action"
			else:
				self.episode_status = "ended"
			self.restart_delay = self.game_settings.delay_after_restart
		return actions_map['reset']
		
		# See if we are in the restart-delaying state
		if self.restart_delay > 0:
			print "Restart delay:", self.restart_delay
			self.restart_delay -= 1
			return actions_map['player_a_noop']
		# See if we should apply the very first action
		if self.episode_status == "first_action":
			print "Sending first action:", self.game_settings.first_action
			self.episode_status = 'ended'
			return actions_map[self.game_settings.first_action]
		terminal = 0
		# See if we are the end of the game
		if self.game_settings.is_end_of_game(screen_matrix, console_ram):
			terminal = 1
			# End the current episode and send a Reset command
			print "End of the game. Restarting."
			if self.game_settings.first_action is not None:
				self.episode_status = "first_action"
			else:
				self.episode_status = "ended"
			self.restart_delay = self.game_settings.delay_after_restart
			return actions_map['reset']

		if reward is None:
			reward = self.game_settings.get_reward(screen_matrix, console_ram)
		self.episode_reward += reward

		if self.episode_status == 'ended':
			print "Episode #%d: Sum Reward = %f" %(self.episode_counter, self.episode_reward)
			self.episode_counter += 1
			self.episode_reward = 0
			self.episode_status = 'started'
			self.save_next_frame = False

		#self.export_image(
		# screen_matrix,
		# 'playing_with_q/episode1/' + str(self.frame_count) + '.png')

		colorScreenMatrix = generalUtils.atari_to_color(self.color_map, screen_matrix)

		#self.export_color_image(
		# colorScreenMatrix,
		# 'deepmind/' + str(self.frame_count) + '.png')

		grayscale = generalUtils.atari_to_grayscale(self.color_map, screen_matrix)
		resized_screen = imresize(grayscale, (83, 83))# - self.background
		resized_screen /= 255.

		if self.save_next_frame:
			self.database.add_episode(curr_state = self.prev_state, next_state = resized_screen, action =
				self.prev_action, reward = 0, terminal = 0)
			self.save_next_frame = False

		self.frame_count = self.frame_count + 1
		print 'Frame: ' + str(self.frame_count) + ', reward in this frame: ' + str(self.episode_reward)
		
		self.add_to_batch(resized_screen)
		
		if self.uniform_over_game_space:
			distribution = stats.rv_discrete(values = (np.arange(6), [0.1, 0.8, 0.1]))
			act_ind = distribution.rvs()
			if act_ind != 1 or np.random.uniform(high = 0.8) < 0.1:
				self.save_next_frame = True
		elif self.greedy_action_selection:
			act_ind = self.get_action(self.batch)
			#print self.get_influence(tmp_batch)
			#if self.frame_count == 64:
			# for i in range(16):
			# plt.imshow(self.get_outputs(tmp_batch, i), interpolation='nearest')
			# plt.show()

			#plt.show()'''
			#if self.frame_count == 100:
			#plt.imshow(self.get_influence(tmp_batch)[0][act_ind], interpolation='nearest')
			#for i in range(256):
			#print i
			#plt.imshow(self.influence[i], interpolation='nearest')
			#plt.show()
			#plt.imshow(self.get_influence(tmp_batch)[0][0], interpolation='nearest')
			#plt.show()
		else:
			values = (np.arange(6), self.get_probs(self.batch))
			distribution = stats.rv_discrete(values = values)
			print values
			#if np.random.uniform(high = 1.0) < 0.9:
			# act_ind = 4
			#else:
			act_ind = distribution.rvs()

		if self.plot_histogram:

			fig = plt.figure()

			fig_0 = fig.add_subplot(1,2,1)
			fig_0.imshow(colorScreenMatrix, interpolation='nearest')
			fig_0.axes.get_xaxis().set_visible(False)
			fig_0.axes.get_yaxis().set_visible(False)
			#fig_1 = fig.add_subplot(2,2,2)
			#fig_1.imshow(resized_screen, interpolation='nearest', cmap = plt.get_cmap('gray'))

			#fig_2 = fig.add_subplot(2,2,3)
			#fig_2.imshow(self.get_influence(tmp_batch, act_ind)[0][0], interpolation='nearest')
			#print self.get_probs(tmp_batch)
			fig_3 = fig.add_subplot(1,2,2)
			fig_3.barh(np.arange(len(self.game_settings.possible_actions)), self.get_probs(self.batch))

			fig_3.set_yticks(np.arange(len(self.game_settings.possible_actions)) + 0.4)

			fig_3.set_yticklabels(self.game_settings.possible_actions )
			fig.tight_layout()
			#self.influence[0] += self.get_influence(tmp_batch, 0)[0][0]
			#plt.imshow(self.influence[0], interpolation='nearest')
			if not exists('artificial_gameplay'):
				mkdir('artificial_gameplay')
			plt.savefig('artificial_gameplay/with_hist_' + str(self.frame_count) + '.png')
		self.prev_state = resized_screen
		self.prev_action = act_ind

		#with open("playing_with_q/episode1_actions.txt", "a") as myfile:
		# myfile.write(str(act_ind) + '\n')
		new_act = actions_map[self.game_settings.possible_actions[act_ind]]
		return new_act

def run_deepmind_agent(game_settings, which_network, plot_histogram, non_linearity, play_for_frames,
	action_continuity_count, working_directory,
	save_reward_history, plot_reward_history):
	"Runs A.L.E, and collects the specified number of ransom samples"
	if not exists(working_directory):
		mkdir(working_directory)
	player_agent = DeepMindAgent(game_settings, which_network, plot_histogram, non_linearity,
		play_for_frames, action_continuity_count,
		0.05, working_directory)

	run_ale(player_agent, game_settings, working_directory,
		save_reward_history, plot_reward_history)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generates gameplay from a given network')
	parser.add_argument('-g','--game', help='Which game', required=True)
	parser.add_argument('-n','--network', help='Network location', required=True)
	parser.add_argument('-a', action='store_true', default=False, help = 'Plot histogram next to frame')
	parser.add_argument('-l','--non_linearity', help='tanh or ReLU', required=True)

	args = vars(parser.parse_args())

	game_settings = generalUtils.get_game_settings(args['game'])

	game_settings.uses_screen_matrix = True
	save_reward_history = True
	plot_reward_history = True
	working_directory = "./"
	play_for_frames = 1000
	action_continuity_count = 0
	run_deepmind_agent(
		game_settings,
		args['network'],
		args['a'],
		args['non_linearity'],
		play_for_frames,
		action_continuity_count,
		working_directory,
		save_reward_history,
		plot_reward_history)