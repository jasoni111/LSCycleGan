import tensorflow as tf

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# def discriminator_upscale_loss(real, generated,cycled,same):
#   real_loss = loss_obj(tf.ones_like(real), real)
#   zeros = tf.zeros_like(generated)
#   generated_loss = loss_obj(zeros, generated)*3
#   generated_loss += loss_obj(zeros, cycled)
#   generated_loss += loss_obj(zeros, same)
#   total_disc_loss = real_loss + generated_loss
#   return total_disc_loss * 0.5


# def discriminator_loss(real, generated):
#   real_loss = loss_obj(tf.ones_like(real), real)
#   generated_loss = loss_obj(tf.zeros_like(generated), generated)
#   total_disc_loss = real_loss + generated_loss
#   return total_disc_loss * 0.5

# def generator_loss(generated):
#   return loss_obj(tf.ones_like(generated), generated)

def cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return loss1


def identity_loss(real_image, same_image):
  """[summary] 
  This try to maintain the same image if the domain are the same
  Arguments:
      real_image {[type]} -- [description]
      same_image {[type]} -- [description]

  Returns:
      [type] -- [description]
  """
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return 0.5 * loss


def mse_loss(real_image,fake_image):
  loss = tf.reduce_mean(tf.math.square(real_image - fake_image))
  return 0.5 * loss