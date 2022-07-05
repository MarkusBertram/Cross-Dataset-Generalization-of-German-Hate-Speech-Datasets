# model
# forward
# loss

import torch.nn as nn
#from functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

# TRAINING:

# for epoch in xrange(n_epoch):

#     len_dataloader = min(len(dataloader_source), len(dataloader_target))
#     data_source_iter = iter(dataloader_source)
#     data_target_iter = iter(dataloader_target)

#     i = 0
#     while i < len_dataloader:

#         p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
#         alpha = 2. / (1. + np.exp(-10 * p)) - 1

#         # training model using source data
#         data_source = data_source_iter.next()
#         s_img, s_label = data_source

#         my_net.zero_grad()
#         batch_size = len(s_label)

#         input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
#         class_label = torch.LongTensor(batch_size)
#         domain_label = torch.zeros(batch_size)
#         domain_label = domain_label.long()

#         if cuda:
#             s_img = s_img.cuda()
#             s_label = s_label.cuda()
#             input_img = input_img.cuda()
#             class_label = class_label.cuda()
#             domain_label = domain_label.cuda()

#         input_img.resize_as_(s_img).copy_(s_img)
#         class_label.resize_as_(s_label).copy_(s_label)

#         class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
#         err_s_label = loss_class(class_output, class_label)
#         err_s_domain = loss_domain(domain_output, domain_label)

#         # training model using target data
#         data_target = data_target_iter.next()
#         t_img, _ = data_target

#         batch_size = len(t_img)

#         input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
#         domain_label = torch.ones(batch_size)
#         domain_label = domain_label.long()

#         if cuda:
#             t_img = t_img.cuda()
#             input_img = input_img.cuda()
#             domain_label = domain_label.cuda()

#         input_img.resize_as_(t_img).copy_(t_img)

#         _, domain_output = my_net(input_data=input_img, alpha=alpha)
#         err_t_domain = loss_domain(domain_output, domain_label)
#         err = err_t_domain + err_s_domain + err_s_label
#         err.backward()
#         optimizer.step()

#         i += 1

#         print 'epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
#               % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
#                  err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy())

#     torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
#     test(source_dataset_name, epoch)
#     test(target_dataset_name, epoch)

