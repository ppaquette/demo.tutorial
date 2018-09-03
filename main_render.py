import os
import gym

# Creating mujoco env
env = gym.make('Humanoid-v2')
env.reset()
print('OK ... 1 ... Done resetting env\n')

nb_steps = 0
done = False
while not done:
	obs, rew, done, info = env.step(env.action_space.sample())
	nb_steps += 1
print('OK ... 2 ... Done after %d steps...\n' % nb_steps)

# Trying to render
render_img = env.render(mode='rgb_array')
print('Got shape:', render_img.shape)
print('OK ... 3 ...  Rendered successfully\n')

# Trying to use TF
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

with tf.Session() as sess:
	result = sess.run(c)
	print('OK ... 4 ... Got TF result\n', result)
print('OK... 5 ... Done testing TF')
