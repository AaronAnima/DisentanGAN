import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_cat2dog_train
from models import get_D, get_Ec, get_Ea, get_G, get_G_zc, get_E_x2zc, get_E_x2za, get_D_content
import random
import argparse
import math
import scipy.stats as stats
import sys
import tensorflow_probability as tfp
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
temp_out = sys.stdout

parser = argparse.ArgumentParser()
parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
args = parser.parse_args()


# def KL_loss(logits):
#     logits_m = tf.reduce_mean(logits, axis=0)
#     # axis = 0, then calc mean along the "vec_ele" with each batch
#     logits_var = tf.reduce_mean((logits - logits_m) * (logits - logits_m), axis=0)
#     logits_sigma = tf.math.sqrt(logits_var)
#     sum_logits = 0.5 * logits_m * logits_m + 0.5 * logits_sigma * logits_sigma - tf.math.log(logits_sigma)
#     kld = logits.shape[1]/2.0 + tf.reduce_mean(sum_logits)
#     return kld


def KL_loss(app_vec_mu, app_vec_logvar):
    KL_element = (pow(app_vec_mu, 2) + np.exp(app_vec_logvar)) * (-1) + 1 + app_vec_logvar
    KL_loss = np.sum(KL_element) * (-0.5)
    return KL_loss


def train(con=False):
    dataset = get_cat2dog_train()
    len_dataset = flags.len_dataset
    E_x_a = get_Ea([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E_x_c = get_Ec([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E_y_a = get_Ea([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E_y_c = get_Ec([None, flags.img_size_h, flags.img_size_w, flags.c_dim])

    G_x = get_G([None, flags.za_dim], [None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]])
    G_y = get_G([None, flags.za_dim], [None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]])
    G_z_c = get_G_zc([None, flags.zc_dim])

    D_x = get_D([None, flags.img_size_h, flags.img_size_h, flags.c_dim])
    D_y = get_D([None, flags.img_size_h, flags.img_size_h, flags.c_dim])
    D_c = get_D_content([None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]])

    E_y_zc = get_E_x2zc([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E_y_za = get_E_x2za([None, flags.img_size_h, flags.img_size_w, flags.c_dim])

    if con:
        E_x_a.load_weights('./checkpoint/E_x_a.npz')
        E_x_c.load_weights('./checkpoint/E_x_c.npz')
        E_y_a.load_weights('./checkpoint/E_y_a.npz')
        E_y_c.load_weights('./checkpoint/E_y_c.npz')
        G_x.load_weights('./checkpoint/G_x.npz')
        G_y.load_weights('./checkpoint/G_y.npz')
        G_z_c.load_weights('./checkpoint/G_z_c.npz')
        D_x.load_weights('./checkpoint/D_x.npz')
        D_y.load_weights('./checkpoint/D_y.npz')
        D_c.load_weights('./checkpoint/D_c.npz')
        E_y_zc.load_weights('./checkpoint/E_y_zc.npz')
        E_y_za.load_weights('./checkpoint/E_y_za.npz')

    E_x_a.train()
    E_x_c.train()
    E_y_a.train()
    E_y_c.train()
    G_x.train()
    G_y.train()
    G_z_c.train()
    E_y_zc.train()
    E_y_za.train()
    D_x.train()
    D_y.train()
    D_c.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = flags.n_epoch

    lr_share = flags.lr

    E_x_a_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    E_x_c_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    E_y_a_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    E_y_c_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    G_x_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    G_y_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    G_z_c_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    E_y_zc_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    E_y_za_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    D_x_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    D_y_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    D_c_optimizer = tf.optimizers.Adam(lr_share, beta_1=flags.beta1, beta_2=flags.beta2)
    tfd = tfp.distributions
    dist = tfd.Normal(loc=0., scale=1.)

    for step, cat_and_dog in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        cat_img = cat_and_dog[0]  # (1, 256, 256, 3)
        dog_img = cat_and_dog[1]  # (1, 256, 256, 3)

        epoch_num = step // n_step_epoch

        with tf.GradientTape(persistent=True) as tape:
            z_a = dist.sample([flags.batch_size_train, flags.za_dim])
            z_c = dist.sample([flags.batch_size_train, flags.zc_dim])

            # dog_app_vec = E_x_a(dog_img)
            dog_app_vec, dog_app_mu, dog_app_logvar = E_x_a(dog_img)  # instead of dog_app_vec

            dog_cont_vec = E_x_c(dog_img)

            z_cat_cont_vec = G_z_c(z_c)

            dog_cont_vec_logit = D_c(dog_cont_vec)
            z_cat_cont_vec_logit = D_c(z_cat_cont_vec)
            # print(dog_app_vec.shape)  # (1, 8)
            # print(z_cat_cont_vec.shape)  # (1, 64, 64, 256)
            fake_dog = G_x([dog_app_vec, z_cat_cont_vec])
            fake_cat = G_y([z_a, dog_cont_vec])

            real_dog_logit = D_x(dog_img)
            fake_dog_logit = D_x(fake_dog)
            real_cat_logit = D_y(cat_img)
            fake_cat_logit = D_y(fake_cat)

            # fake_dog_app_vec = E_x_a(fake_dog)
            fake_dog_app_vec, fake_dog_app_mu, fake_dog_app_logvar = E_x_a(fake_dog)  # instead of dog_app_vec

            fake_dog_cont_vec = E_x_c(fake_dog)
            # fake_cat_app_vec = E_y_a(fake_cat)
            fake_cat_app_vec, fake_cat_app_mu, fake_cat_app_logvar = E_x_a(fake_cat)  # instead of dog_app_vec

            fake_cat_cont_vec = E_y_c(fake_cat)

            recon_dog = G_x([fake_dog_app_vec, fake_cat_cont_vec])
            recon_cat = G_y([fake_cat_app_vec, fake_dog_cont_vec])

            recon_z_a = E_y_za(recon_cat)
            recon_z_c = E_y_zc(recon_cat)

            # content adv loss, to update D_c E_x_c, E_y_c
            cont_adv_loss = flags.lambda_content * (
                        1 / 2 * tl.cost.sigmoid_cross_entropy(dog_cont_vec_logit, tf.ones_like(dog_cont_vec_logit)) + \
                        1 / 2 * tl.cost.sigmoid_cross_entropy(dog_cont_vec_logit, tf.zeros_like(dog_cont_vec_logit)) + \
                        1 / 2 * tl.cost.sigmoid_cross_entropy(z_cat_cont_vec_logit,
                                                              tf.zeros_like(z_cat_cont_vec_logit)) + \
                        1 / 2 * tl.cost.sigmoid_cross_entropy(z_cat_cont_vec_logit, tf.ones_like(z_cat_cont_vec_logit)))

            # cross_identity loss, to update all Es and Gs
            cross_identity_loss = flags.lambda_corss * (
                        tl.cost.absolute_difference_error(recon_dog, dog_img, is_mean=True) + \
                        tl.cost.absolute_difference_error(recon_z_a, z_a, is_mean=True) + \
                        tl.cost.absolute_difference_error(recon_z_c, z_c, is_mean=True))

            # Domain adv loss
            dog_adv_loss = flags.lambda_domain * (
                        tl.cost.sigmoid_cross_entropy(real_dog_logit, tf.ones_like(real_dog_logit)) + \
                        tl.cost.sigmoid_cross_entropy(fake_dog_logit, tf.zeros_like(fake_dog_logit)))
            cat_adv_loss = flags.lambda_domain * (
                        tl.cost.sigmoid_cross_entropy(real_cat_logit, tf.ones_like(real_cat_logit)) + \
                        tl.cost.sigmoid_cross_entropy(fake_cat_logit, tf.zeros_like(fake_cat_logit)))

            # Self recon loss
            self_recon_dog = G_x([dog_app_vec, dog_cont_vec])
            self_recon_cat_z = G_y([E_y_za(cat_img), G_z_c(E_y_zc(cat_img))])
            self_recon_cat_t = G_y([E_y_a(cat_img)[0], E_y_c(cat_img)])

            cat_self_recon_loss_z = flags.lambda_srecon * tl.cost.absolute_difference_error(self_recon_cat_z, cat_img,
                                                                                            is_mean=True)
            cat_self_recon_loss_t = flags.lambda_srecon * tl.cost.absolute_difference_error(self_recon_cat_t, cat_img,
                                                                                            is_mean=True)
            dog_self_recon_loss = flags.lambda_srecon * tl.cost.absolute_difference_error(self_recon_dog, dog_img,
                                                                                          is_mean=True)

            # latent regression loss
            fake_cat_za = E_y_za(fake_cat)
            z_a_dog = dist.sample([flags.batch_size_train, flags.za_dim])
            fake_dog_za = E_x_a(G_x([z_a_dog, E_x_c(dog_img)]))[0]

            latent_regre_loss_cat = tl.cost.absolute_difference_error(fake_cat_za, z_a, is_mean=True)
            latent_regre_loss_dog = tl.cost.absolute_difference_error(fake_dog_za, z_a_dog, is_mean=True)

            # latent generation loss
            gen_cat = G_y([z_a, G_z_c(z_c)])
            gen_cat_logit = D_y(gen_cat)

            latent_gen_loss_cat = flags.lambda_latent * (
                        tl.cost.sigmoid_cross_entropy(real_cat_logit, tf.ones_like(real_cat_logit)) + \
                        tl.cost.sigmoid_cross_entropy(gen_cat_logit, tf.zeros_like(gen_cat_logit)))

            # KL loss
            z_kl = dist.sample([flags.KL_batch, flags.za_dim])
            # kl_mean, kl_variance = tf.nn.moments(x=dog_app_vec, axes=[1])
            # print(dog_app_vec)
            # print(str(kl_variance) + ' ' + str(kl_mean)) # [1,] [1,]
            # kl_sigma = tf.math.sqrt(kl_variance)
            # z_calc = z_kl * kl_sigma + tf.ones_like(z_kl) * kl_mean
            kl_loss_dog = flags.lambda_KL * KL_loss(dog_app_mu, dog_app_logvar)

            # Mode seeking regularization
            z_1 = dist.sample([flags.batch_size_train, flags.za_dim])
            z_2 = dist.sample([flags.batch_size_train, flags.za_dim])
            dog_1 = G_x([z_1, E_x_c(dog_img)])
            dog_2 = G_x([z_2, E_x_c(dog_img)])
            dog_norm = tl.cost.mean_squared_error(dog_1, dog_2)
            cat_norm = tl.cost.mean_squared_error(G_y([z_1, E_y_c(cat_img)]), G_y([z_2, E_y_c(cat_img)]))
            z_norm = tl.cost.mean_squared_error(z_1, z_2)

            dog_ms_loss = - dog_norm / z_norm
            cat_ms_loss = - cat_norm / z_norm

            # sum up total loss
            content_adv_loss = cont_adv_loss
            cross_identity_loss = cross_identity_loss
            domain_adv_loss = dog_adv_loss + cat_adv_loss
            self_recon_loss = cat_self_recon_loss_t + cat_self_recon_loss_z + dog_self_recon_loss
            latent_regre_loss = latent_gen_loss_cat + latent_regre_loss_dog
            latent_gen_loss = latent_gen_loss_cat
            kl_loss = kl_loss_dog
            ms_loss = dog_ms_loss + cat_ms_loss

            E_x_a_total_loss = cross_identity_loss + dog_self_recon_loss + latent_regre_loss_dog + kl_loss_dog
            E_x_c_total_loss = cont_adv_loss + cross_identity_loss + dog_self_recon_loss + latent_regre_loss_dog + \
                               dog_ms_loss
            E_y_a_total_loss = cross_identity_loss + cat_self_recon_loss_t + latent_regre_loss_cat
            E_y_c_total_loss = cont_adv_loss + cross_identity_loss + cat_self_recon_loss_t + latent_regre_loss_cat + \
                               cat_ms_loss
            E_y_zc_total_loss = cross_identity_loss + cat_self_recon_loss_z
            E_y_za_total_loss = cross_identity_loss + cat_self_recon_loss_z

            G_x_total_loss = cross_identity_loss + dog_adv_loss + dog_self_recon_loss + latent_regre_loss_dog + \
                             dog_ms_loss
            G_y_total_loss = cross_identity_loss + cat_adv_loss + cat_self_recon_loss_z + cat_self_recon_loss_t + \
                             latent_regre_loss_cat + latent_gen_loss_cat + cat_ms_loss
            G_z_c_total_loss = cross_identity_loss + cat_self_recon_loss_z

            D_x_total_loss = dog_adv_loss
            D_y_total_loss = cat_adv_loss + latent_gen_loss_cat
            D_c_total_loss = cont_adv_loss

        # Release Memory
        E_x_a.release_memory()
        E_x_c.release_memory()
        E_y_a.release_memory()
        E_y_c.release_memory()
        E_y_zc.release_memory()
        E_y_za.release_memory()
        G_x.release_memory()
        G_y.release_memory()
        G_z_c.release_memory()
        D_x.release_memory()
        D_y.release_memory()
        D_c.release_memory()
        # Updating Encoder
        grad = tape.gradient(E_x_a_total_loss, E_x_a.trainable_weights)
        E_x_a_optimizer.apply_gradients(zip(grad, E_x_a.trainable_weights))

        grad = tape.gradient(E_x_c_total_loss, E_x_c.trainable_weights)
        E_x_c_optimizer.apply_gradients(zip(grad, E_x_c.trainable_weights))

        grad = tape.gradient(E_y_a_total_loss, E_y_a.trainable_weights)
        E_y_a_optimizer.apply_gradients(zip(grad, E_y_a.trainable_weights))

        grad = tape.gradient(E_y_c_total_loss, E_y_c.trainable_weights)
        E_y_c_optimizer.apply_gradients(zip(grad, E_y_c.trainable_weights))

        grad = tape.gradient(E_y_zc_total_loss, E_y_zc.trainable_weights)
        E_y_zc_optimizer.apply_gradients(zip(grad, E_y_zc.trainable_weights))

        grad = tape.gradient(E_y_za_total_loss, E_y_za.trainable_weights)
        E_y_za_optimizer.apply_gradients(zip(grad, E_y_za.trainable_weights))

        grad = tape.gradient(G_x_total_loss, G_x.trainable_weights)
        G_x_optimizer.apply_gradients(zip(grad, G_x.trainable_weights))

        grad = tape.gradient(G_y_total_loss, G_y.trainable_weights)
        G_y_optimizer.apply_gradients(zip(grad, G_y.trainable_weights))

        grad = tape.gradient(G_z_c_total_loss, G_z_c.trainable_weights)
        G_z_c_optimizer.apply_gradients(zip(grad, G_z_c.trainable_weights))

        grad = tape.gradient(D_x_total_loss, D_x.trainable_weights)
        D_x_optimizer.apply_gradients(zip(grad, D_x.trainable_weights))

        grad = tape.gradient(D_y_total_loss, D_y.trainable_weights)
        D_y_optimizer.apply_gradients(zip(grad, D_y.trainable_weights))

        grad = tape.gradient(D_c_total_loss, D_c.trainable_weights)
        D_c_optimizer.apply_gradients(zip(grad, D_c.trainable_weights))

        del tape

        # show current state
        if np.mod(step, flags.show_every_step) == 0:
            with open("log.txt", "a+") as f:
                sys.stdout = f
                print("Epoch: [{}/{}] [{}/{}] content_adv_loss:{:5f}, cross_identity_loss:{:5f}, "
                      "domain_adv_loss:{:5f}, self_recon_loss:{:5f}, latent_regre_loss:{:5f}, latent_gen_loss:{:5f}, "
                      "kl_loss:{:5f}, ms_loss:{:10f}".format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, content_adv_loss,
                       cross_identity_loss, domain_adv_loss, self_recon_loss, latent_regre_loss, latent_gen_loss,
                       kl_loss, ms_loss))

                sys.stdout = temp_out
                print("Epoch: [{}/{}] [{}/{}] content_adv_loss:{:5f}, cross_identity_loss:{:5f}, "
                      "domain_adv_loss:{:5f}, self_recon_loss:{:5f}, latent_regre_loss:{:5f}, latent_gen_loss:{:5f}, "
                      "kl_loss:{:5f}, ms_loss:{:10f}".format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, content_adv_loss,
                       cross_identity_loss, domain_adv_loss, self_recon_loss, latent_regre_loss, latent_gen_loss,
                       kl_loss, ms_loss))

        if np.mod(step, flags.save_step) == 0 and step != 0:
            E_x_a.save_weights('{}/{}/E_x_a.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_x_c.save_weights('{}/{}/E_x_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_y_a.save_weights('{}/{}/E_y_a.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_y_c.save_weights('{}/{}/E_y_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            G_x.save_weights('{}/{}/G_x.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            G_y.save_weights('{}/{}/G_y.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            G_z_c.save_weights('{}/{}/G_z_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_y_zc.save_weights('{}/{}/E_y_zc.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E_y_za.save_weights('{}/{}/E_y_za.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_x.save_weights('{}/{}/D_x.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_y.save_weights('{}/{}/D_y.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_c.save_weights('{}/{}/D_c.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')

            # G.train()

        if np.mod(step, flags.eval_step) == 0:
            z = dist.sample([flags.batch_size_train, flags.za_dim])
            E_y_c.eval()
            G_y.eval()
            eval_cat_cont_vec = E_y_c(cat_img)
            sys_cat_img = G_y([z, eval_cat_cont_vec])
            sys_cat_img = tf.concat([sys_cat_img, cat_img], 0)
            E_y_c.train()
            G_y.train()
            tl.visualize.save_images(sys_cat_img.numpy(), [1, 2],
                                     '{}/{}/train_{:02d}_{:04d}.png'.format(flags.sample_dir, flags.param_dir,
                                                                            step // n_step_epoch, step))



if __name__ == '__main__':
    # To choose flags

    # To make sure path is legal

    # Start training process

    tl.files.exists_or_mkdir(flags.checkpoint_dir + '/' + flags.param_dir)  # checkpoint path
    tl.files.exists_or_mkdir(flags.sample_dir + '/' + flags.param_dir)  # samples path
    train(con=args.is_continue)
