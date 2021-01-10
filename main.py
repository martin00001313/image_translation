import core
import datetime
import os
import tensorflow as tf

from matplotlib import pyplot as plt

BASE_PATH = '/home/martin/Desktop/arm/'

def main_fn():
    train_dataset, test_dataset = core.get_data()

    generator = core.Generator()

    # check descriminator model
    discriminator = core.Discriminator()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = BASE_PATH + 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    for example_input, example_target in test_dataset.take(0):
        core.generate_images(generator, example_input, example_target)

    summary_writer = tf.summary.create_file_writer(
        BASE_PATH + 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    core.fit(train_dataset, core.EPOCHS, test_dataset, generator, checkpoint, checkpoint_prefix, discriminator,
             generator_optimizer, discriminator_optimizer, summary_writer, loss_object)

    plt.show()


if __name__ == "__main__":
    main_fn()