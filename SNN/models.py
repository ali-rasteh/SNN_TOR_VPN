import torch
import numpy as np
from math import sqrt
from math import pow


isConvlayer = lambda l : isinstance(l, SpikingConv3DLayer) or isinstance(l, SpikingConv2DLayer_custom) or isinstance(l, SpikingConv2DLayer) or isinstance(l, SpikingConv1DLayer)
isSeperableConvlayer = lambda l : isinstance(l, SpikingConv3DLayer_separable)
isPoolinglayer = lambda l : isinstance(l, Spiking3DPoolingLayer)
isDropoutlayer = lambda l : isinstance(l, DropoutLayer)
isBatchNormLayer = lambda l : isinstance(l, BatchNormLayer)
isReadoutlayer = lambda l : isinstance(l, ReadoutLayer)
layerHasParamsW = lambda l : not (isinstance(l, Spiking3DPoolingLayer) or isinstance(l,DropoutLayer) or isinstance(l,BatchNormLayer))
layerHasParamsV = lambda l : not (isinstance(l, Spiking3DPoolingLayer) or isinstance(l,DropoutLayer) or isinstance(l,BatchNormLayer))
layerHasParamsB = lambda l : not (isinstance(l, Spiking3DPoolingLayer) or isinstance(l,DropoutLayer) or isinstance(l,BatchNormLayer))
layerHasParamsBeta = lambda l : not (isinstance(l, Spiking3DPoolingLayer) or isinstance(l,DropoutLayer) or isinstance(l,BatchNormLayer))
layerHasHeaviside = lambda l : not (isinstance(l, Spiking3DPoolingLayer) or isinstance(l,DropoutLayer) or isinstance(l,BatchNormLayer))



class SNN(torch.nn.Module):

    def __init__(self, layers):

        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        loss_seq = []

        with torch.autograd.profiler.record_function("label-forward"):
            for l in self.layers:
                x, loss = l(x)
                loss_seq.append(loss)

        return x, loss_seq


    def clamp(self):
        for l in self.layers:
            l.clamp()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()


class SpikingDenseLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape, spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params,
                 recurrent=False, lateral_connections=True, Regularization_Term=["squared", "max"],
                 High_speed_mode = False, eps=1e-8):

        super(SpikingDenseLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term
        self.High_speed_mode = High_speed_mode

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode']=='all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode'] == 'one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]
        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # membrane potential
        mem = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)

        if not self.High_speed_mode:
            if self.lateral_connections:
                d = torch.einsum("ab, ac -> bc", self.w, self.w)

            with torch.autograd.profiler.record_function("label-sum"):
                norm = (self.w**2).sum(0)

        Max_spikes_per_run_mat = self.Max_spikes_per_run * torch.ones_like(mem)
        Previous_spikes = torch.zeros_like(spk)

        for t in range(nb_steps):

            # reset term
            if self.High_speed_mode:
                rst = torch.zeros_like(mem)
                c = ((mem - 1) > 0)
                rst[c] = torch.ones_like(mem)[c]
                mem_tmp = mem
            else :
                if self.lateral_connections:
                    rst = torch.einsum("ab,bc ->ac", spk, d)
                else:
                    rst = spk*self.b*norm

            input_ = h[:,t,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # membrane potential update
            if(self.Leaky_Neuron) :
                if self.spike_limit_mode=="soft_reset":
                    mem = (mem-rst)*self.beta + input_*(1.-self.beta)
                elif self.spike_limit_mode=="hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_*(1.-self.beta)
            else :
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_

            if self.High_speed_mode:
                mthr = mem_tmp - 1
            else :
                mthr = torch.einsum("ab,b->ab",mem, 1./(norm+self.eps))-self.b

            if (self.spike_limit):
                spk = (Max_spikes_per_run_mat - Previous_spikes) * self.spike_fn(mthr, self.surrogate_sigma)
                Previous_spikes+=spk
            else :
                spk = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec[:,t,:] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if(self.Regularization_Term[0] == "squared") :
            loss = 0.5*(spk_rec**2).mean()
        elif(self.Regularization_Term[0] == "simple") :
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,2)).max()/(batch_size*self.output_shape)
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,2)))**2).mean()/(batch_size*self.output_shape)
        loss = [loss,spk_spread_loss]

        return spk_rec, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.output_shape))

        if self.train_params['beta_init_method']=="constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method']=="normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'], std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method']=="uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'], b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method']=="constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method']=="normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'], std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method']=="uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)


class SpikingConv1DLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, recurrent=False,
                 lateral_connections=True,
                 eps=1e-8, stride=1,flatten_output=False):

        super(SpikingConv1DLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv1d(x, self.w, padding=(np.ceil(((self.kernel_size-1)*self.dilation)/2).astype(int),),
                                      dilation=(self.dilation,),
                                      stride=(self.stride,))
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels),  dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps),  dtype=x.dtype, device=x.device)


        if self.lateral_connections:
            d = torch.einsum("abc, ebc -> ae", self.w, self.w)
        b = self.b

        with torch.autograd.profiler.record_function("label-sum"):
            norm = (self.w**2).sum((1,2))


        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bd ->ad", spk, d)
            else:
                rst = torch.einsum("ab,b->ab", spk, self.b*norm)

            input_ = conv_x[:,:,t]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bd->ad", spk, self.v)

            # membrane potential update
            mem = (mem-rst)*self.beta + input_*(1.-self.beta)
            mthr = torch.einsum("ab,b->ab",mem, 1./(norm+self.eps))-b

            spk = self.spike_fn(mthr)

            spk_rec[:,:,t] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        loss = 0.5*(spk_rec**2).mean()

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):

        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)


class SpikingConv2DLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"],
                 eps=1e-8, stride=(1,1),flatten_output=False):

        super(SpikingConv2DLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term

        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.out_channels, self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode'] == 'one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(np.ceil(((np.array(self.kernel_size)-1)*np.array(self.dilation))/2).astype(int)),
                                      dilation=self.dilation,
                                      stride=self.stride)
        conv_x = conv_x[:,:,:,:self.output_shape]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels, self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, self.output_shape),  dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)


        if self.lateral_connections:
            d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)
        b = self.b.unsqueeze(1).repeat((1,self.output_shape))

        with torch.autograd.profiler.record_function("label-sum"):
            norm = (self.w**2).sum((1,2,3))

        Max_spikes_per_run_mat = self.Max_spikes_per_run * torch.ones_like(mem)
        Previous_spikes = torch.zeros_like(spk)

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abc,bd ->adc", spk, d)
            else:
                rst = torch.einsum("abc,b->abc", spk, self.b*norm)

            input_ = conv_x[:,:,t,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("abc,bd->adc", spk, self.v)

            # membrane potential update
            if (self.Leaky_Neuron):
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem-rst)*self.beta + input_*(1.-self.beta)
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_ * (1. - self.beta)
            else:
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_

            mthr = torch.einsum("abc,b->abc",mem, 1./(norm+self.eps))-b

            if (self.spike_limit):
                spk = (Max_spikes_per_run_mat - Previous_spikes) * self.spike_fn(mthr, self.surrogate_sigma)
                Previous_spikes+=spk
            else :
                spk = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec[:,:,t,:] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5*(spk_rec**2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3)).max()/(batch_size*self.out_channels*self.output_shape)
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3)))**2).mean()/(batch_size*self.out_channels*self.output_shape)
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels*self.output_shape)

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)


class SpikingConv2DLayer_custom(torch.nn.Module):

    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"],
                 High_speed_mode=False, eps=1e-8, stride=(1,1),flatten_output=False):

        super(SpikingConv2DLayer_custom, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.nb_steps = spike_params["nb_steps"]
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term
        self.High_speed_mode = High_speed_mode

        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, *output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.out_channels, *self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode'] == 'one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        self.mem_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(np.ceil(((np.array(self.kernel_size)-1)*np.array(self.dilation))/2).astype(int)),
                                            dilation=self.dilation,
                                            stride=self.stride)
        conv_x = conv_x[:, :, :self.output_shape[0], :self.output_shape[1]]
        nb_steps = self.nb_steps

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels, *self.output_shape), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, *self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, *self.output_shape), dtype=x.dtype,
                              device=x.device)
        mem_rec = torch.zeros((batch_size, self.out_channels, nb_steps, *self.output_shape), dtype=x.dtype,
                              device=x.device)

        if not self.High_speed_mode:
            if self.lateral_connections:
                d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)
            b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

            with torch.autograd.profiler.record_function("label-sum"):
                norm = (self.w ** 2).sum((1, 2, 3))

        Max_spikes_per_run_mat = self.Max_spikes_per_run * torch.ones_like(mem)
        Previous_spikes = torch.zeros_like(spk)
        input_ = conv_x

        for t in range(nb_steps):

            # reset term
            if self.High_speed_mode:
                rst = torch.zeros_like(mem)
                c = ((mem - 1) > 0)
                rst[c] = torch.ones_like(mem)[c]
                mem_tmp = mem
            else :
                if self.lateral_connections:
                    rst = torch.einsum("abcd,be ->aecd", spk, d)
                else:
                    rst = torch.einsum("abcd,b->abcd", spk, self.b * norm)

            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            # membrane potential update
            if (self.Leaky_Neuron):
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_ * (1. - self.beta)
            else:
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_

            if self.High_speed_mode:
                mthr = mem_tmp - 1
            else :
                mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + self.eps)) - b

            if (self.spike_limit):
                spk = (Max_spikes_per_run_mat - Previous_spikes) * self.spike_fn(mthr, self.surrogate_sigma)
                Previous_spikes+=spk
            else:
                spk = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec[:, :, t, :, :] = spk
            mem_rec[:, :, t, :, :] = mem

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5 * (spk_rec ** 2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3,4)).max()/(batch_size*self.out_channels*np.prod(self.output_shape))
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3,4)))**2).mean()/(batch_size*self.out_channels*np.prod(self.output_shape))
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)


class SpikingConv2DLayer_custom_fast(torch.nn.Module):
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"],
                 High_speed_mode=False, eps=1e-8, stride=(1, 1), flatten_output=False):

        super(SpikingConv2DLayer_custom_fast, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.nb_steps = spike_params["nb_steps"]
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term
        self.High_speed_mode = High_speed_mode

        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, *output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.out_channels, self.nb_steps, *self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode'] == 'one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        self.mem_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(
            np.ceil(((np.array(self.kernel_size) - 1) * np.array(self.dilation)) / 2).astype(int)),
                                            dilation=self.dilation,
                                            stride=self.stride)
        conv_x = conv_x[:, :, :self.output_shape[0], :self.output_shape[1]]
        nb_steps = self.nb_steps
        conv_x = conv_x.unsqueeze(2).repeat(1,1,nb_steps,1,1)

        if not self.Leaky_Neuron and not self.lateral_connections and self.spike_limit:
            b = self.b.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat((1, nb_steps, *self.output_shape))
            with torch.autograd.profiler.record_function("label-sum"):
                norm = (self.w ** 2).sum((1, 2, 3))
            mem = conv_x.cumsum(dim=2)
            mem_rec = mem
            mthr = torch.einsum("abcde,b->abcde", mem, 1. / (norm + self.eps)) - b
            spk_rec_init = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec_init_inverse = 1 - spk_rec_init
            spk_rec_init_inverse_cumprod = spk_rec_init_inverse.cumprod(dim=2)
            spk_rec_init_inverse_cumprod_inverse = 1 - spk_rec_init_inverse_cumprod
            spk_rec_init_inverse_cumprod_inverse_roll = spk_rec_init_inverse_cumprod_inverse.roll(shifts=1, dims=2)
            spk_rec_init_inverse_cumprod_inverse_roll[:, :, 0, :, :] = 0
            spk_rec = spk_rec_init_inverse_cumprod_inverse - spk_rec_init_inverse_cumprod_inverse_roll


        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5 * (spk_rec ** 2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3,4)).max()/(batch_size*self.out_channels*np.prod(self.output_shape))
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3,4)))**2).mean()/(batch_size*self.out_channels*np.prod(self.output_shape))
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.in_channels * np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


class SpikingConv3DLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, groups=1, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"], High_speed_mode = False,
                 eps=1e-8, stride=(1,1,1), flatten_output=False):

        super(SpikingConv3DLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term
        self.High_speed_mode = High_speed_mode

        self.w = torch.nn.Parameter(torch.empty((out_channels, int(in_channels/groups), *kernel_size)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, *output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.out_channels, *self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode']=='one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        with torch.autograd.profiler.record_function("label-conv3d"):
            conv_x = torch.nn.functional.conv3d(x, self.w, padding=tuple(np.ceil(((np.array(self.kernel_size)-1)*np.array(self.dilation))/2).astype(int)),
                                          dilation=self.dilation,
                                        stride=self.stride, groups=self.groups)
            conv_x = conv_x[:,:,:,:self.output_shape[0],:self.output_shape[1]]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels, *self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, *self.output_shape),  dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, *self.output_shape),  dtype=x.dtype, device=x.device)

        if not self.High_speed_mode :
            if self.lateral_connections:
                d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
            b = self.b.unsqueeze(1).unsqueeze(1).repeat((1,*self.output_shape))

            with torch.autograd.profiler.record_function("label-sum"):
                norm = (self.w**2).sum((1,2,3,4))

        Max_spikes_per_run_mat = self.Max_spikes_per_run * torch.ones_like(mem)
        Previous_spikes = torch.zeros_like(spk)

        for t in range(nb_steps):

            # reset term
            if self.High_speed_mode :
                rst = torch.zeros_like(mem)
                c = ((mem - 1) > 0)
                rst[c] = torch.ones_like(mem)[c]
                mem_tmp = mem
            else :
                if self.lateral_connections:
                    rst = torch.einsum("abcd,be ->aecd", spk, d)
                else:
                    rst = torch.einsum("abcd,b->abcd", spk, self.b*norm)


            input_ = conv_x[:,:,t,:,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            # membrane potential update
            if (self.Leaky_Neuron):
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem-rst)*self.beta + input_*(1.-self.beta)
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_ * (1. - self.beta)
            else :
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_

            if self.High_speed_mode:
                mthr = mem_tmp-1
            else :
                mthr = torch.einsum("abcd,b->abcd",mem, 1./(norm+self.eps))-b

            if (self.spike_limit):
                spk = (Max_spikes_per_run_mat - Previous_spikes) * self.spike_fn(mthr, self.surrogate_sigma)
                Previous_spikes+=spk
            else:
                spk = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec[:,:,t,:,:] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5*(spk_rec**2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3,4)).max()/(batch_size*self.out_channels*np.prod(self.output_shape))
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3,4)))**2).mean()/(batch_size*self.out_channels*np.prod(self.output_shape))
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels*np.prod(self.output_shape))

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)


class SpikingConv3DLayer_fast(torch.nn.Module):
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, groups=1, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"], High_speed_mode=False,
                 eps=1e-8, stride=(1, 1, 1), flatten_output=False):

        super(SpikingConv3DLayer_fast, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term
        self.High_speed_mode = High_speed_mode

        self.w = torch.nn.Parameter(torch.empty((out_channels, int(in_channels / groups), *kernel_size)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, *output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.out_channels, self.nb_steps, *self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode']=='one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        with torch.autograd.profiler.record_function("label-conv3d"):
            conv_x = torch.nn.functional.conv3d(x, self.w, padding=tuple(
                np.ceil(((np.array(self.kernel_size) - 1) * np.array(self.dilation)) / 2).astype(int)),
                                                dilation=self.dilation,
                                                stride=self.stride, groups=self.groups)
            conv_x = conv_x[:, :, :, :self.output_shape[0], :self.output_shape[1]]
        nb_steps = conv_x.shape[2]

        if not self.Leaky_Neuron and not self.lateral_connections and self.spike_limit:
            b = self.b.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat((1, nb_steps, *self.output_shape))
            with torch.autograd.profiler.record_function("label-sum"):
                norm = (self.w ** 2).sum((1, 2, 3, 4))
            mem = conv_x.cumsum(dim=2)
            mthr = torch.einsum("abcde,b->abcde", mem, 1. / (norm + self.eps)) - b
            spk_rec_init = self.spike_fn(mthr, self.surrogate_sigma)

            spk_rec_init_inverse = 1 - spk_rec_init
            spk_rec_init_inverse_cumprod = spk_rec_init_inverse.cumprod(dim=2)
            spk_rec_init_inverse_cumprod_inverse = 1 - spk_rec_init_inverse_cumprod
            spk_rec_init_inverse_cumprod_inverse_roll = spk_rec_init_inverse_cumprod_inverse.roll(shifts=1, dims=2)
            spk_rec_init_inverse_cumprod_inverse_roll[:,:,0,:,:] = 0
            spk_rec = spk_rec_init_inverse_cumprod_inverse - spk_rec_init_inverse_cumprod_inverse_roll

            # print(spk_rec_init_inverse_cumprod.shape)
            # spk_rec_init_inverse_cumprod_sum=spk_rec_init_inverse_cumprod.sum(dim=2)
            # print(spk_rec_init_inverse_cumprod_sum.shape)


        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5 * (spk_rec ** 2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3,4)).max()/(batch_size*self.out_channels*np.prod(self.output_shape))
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3,4)))**2).mean()/(batch_size*self.out_channels*np.prod(self.output_shape))
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):

        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.in_channels * np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


class SpikingConv3DLayer_separable(torch.nn.Module):
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, spike_params, train_params, recurrent=False,
                 lateral_connections=True, Regularization_Term=["squared", "max"],
                 eps=1e-8, stride=(1, 1, 1), flatten_output=False):

        super(SpikingConv3DLayer_separable, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3

        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.spike_limit = spike_params["spike_limit"]
        self.spike_limit_mode = spike_params["spike_limit_mode"]
        self.Max_spikes_per_run = spike_params["Max_spikes_per_run"]
        self.Regularization_Term = Regularization_Term

        # Depthwise and Pointwise convolutions
        self.w_dw = torch.nn.Parameter(torch.empty((in_channels, 1, *kernel_size)), requires_grad=train_params['w'])
        self.w_pw = torch.nn.Parameter(torch.empty((out_channels, in_channels, 1, 1, 1)), requires_grad=train_params['w'])
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=train_params['v'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(out_channels, *output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=train_params['b'])

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        with torch.autograd.profiler.record_function("label-conv3d-separable"):
            conv_x = torch.nn.functional.conv3d(x, self.w_dw, padding=tuple(np.ceil(((np.array(self.kernel_size)-1)*np.array(self.dilation))/2).astype(int)),
                                                dilation=self.dilation,
                                                stride=self.stride, groups=self.in_channels)
            conv_x = conv_x[:, :, :, :self.input_shape[0], :self.input_shape[1]]


            conv_x = torch.nn.functional.conv3d(conv_x, self.w_pw, padding=tuple(np.ceil(((np.array(self.kernel_size)-1)*np.array(self.dilation))/2).astype(int)),
                                                dilation=self.dilation,
                                                stride=self.stride, groups=1)
            conv_x = conv_x[:, :, :, :self.output_shape[0], :self.output_shape[1]]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels, *self.output_shape), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, *self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, *self.output_shape), dtype=x.dtype,
                              device=x.device)

        w_dw_tmp = self.w_dw.transpose(0, 1)
        w_dw_tmp = w_dw_tmp.repeat(self.out_channels, 1, 1, 1, 1)
        w_pw_tmp = self.w_pw.repeat(1, 1, *self.kernel_size)
        w_tmp = w_dw_tmp*w_pw_tmp

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", w_tmp, w_tmp)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        with torch.autograd.profiler.record_function("label-sum"):
            norm = (w_tmp ** 2).sum((1, 2, 3, 4))

        Max_spikes_per_run_mat = self.Max_spikes_per_run * torch.ones_like(mem)
        Previous_spikes = torch.zeros_like(spk)

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abcd,be ->aecd", spk, d)
            else:
                rst = torch.einsum("abcd,b->abcd", spk, self.b * norm)

            input_ = conv_x[:, :, t, :, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            # membrane potential update
            if (self.Leaky_Neuron):
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_ * (1. - self.beta)
            else:
                if self.spike_limit_mode == "soft_reset":
                    mem = (mem - rst) * self.beta + input_
                elif self.spike_limit_mode == "hard_reset":
                    mem = mem * self.beta * (1 - spk) + input_

            mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + self.eps)) - b

            if (self.spike_limit):
                spk = (Max_spikes_per_run_mat - Previous_spikes) * self.spike_fn(mthr)
                Previous_spikes+=spk
            else:
                spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :, :] = spk

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if (self.Regularization_Term[0] == "squared"):
            loss = 0.5 * (spk_rec ** 2).mean()
        elif (self.Regularization_Term[0] == "simple"):
            loss = 0.5 * (spk_rec).mean()
        if (self.Regularization_Term[1] == "max"):
            spk_spread_loss = spk_rec.sum(dim=(0,1,3,4)).max()/(batch_size*self.out_channels*np.prod(self.output_shape))
        elif (self.Regularization_Term[1] == "L2"):
            spk_spread_loss = ((spk_rec.sum(dim=(0,1,3,4)))**2).mean()/(batch_size*self.out_channels*np.prod(self.output_shape))
        loss = [loss,spk_spread_loss]

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))

        else:

            output = spk_rec

        return output, loss

    def reset_parameters(self):
        w_mean = sqrt(self.w_init_mean)
        w_std = sqrt(sqrt(pow(self.w_init_mean,2)+pow(self.w_init_std * np.sqrt(1. / (self.in_channels * np.prod(self.kernel_size))), 2))-self.w_init_mean)
        torch.nn.init.normal_(self.w_dw, mean=w_mean, std=w_std)
        torch.nn.init.normal_(self.w_pw, mean=w_mean, std=w_std)
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.out_channels))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):

        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


class Spiking3DPoolingLayer(torch.nn.Module):
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation, spike_params, stride=(1, 1), flatten_output=False):

        super(Spiking3DPoolingLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.flatten_output = flatten_output

        self.mode = "2D"

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        if self.mode=="2D":
            x_cumsum = x.cumsum(dim=2)
            x_sum = x_cumsum.sum(dim=2, keepdim=False)
            x_pool_2D, indices_2D = torch.nn.functional.max_pool2d(x_sum, self.kernel_size, self.stride, padding=0, dilation=self.dilation, return_indices=True, ceil_mode=False)

            flattened_tensor = x.flatten(start_dim=3)
            indices_3D=indices_2D.reshape((batch_size, self.out_channels,1,indices_2D.shape[2],indices_2D.shape[3]))
            indices_3D=indices_3D.repeat(1,1,nb_steps,1,1)
            spk_rec = flattened_tensor.gather(dim=3, index=indices_3D.flatten(start_dim=3)).view_as(indices_3D)

        elif self.mode=="3D":
            x_pool, indices = torch.nn.functional.max_pool3d(x, self.kernel_size, self.stride, padding=0, dilation=self.dilation, return_indices=True, ceil_mode=False)
            x_pool_inverse = 1 - x_pool
            x_pool_inverse_cumprod = x_pool_inverse.cumprod(dim=2)
            x_pool_inverse_cumprod_inverse = 1 - x_pool_inverse_cumprod
            x_pool_inverse_cumprod_inverse_roll = x_pool_inverse_cumprod_inverse.roll(shifts=1, dims=2)
            x_pool_inverse_cumprod_inverse_roll[:,:,0,:,:] = 0
            spk_rec = x_pool_inverse_cumprod_inverse - x_pool_inverse_cumprod_inverse_roll

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        loss = 0 * (spk_rec).mean()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))
        else:
            output = spk_rec

        return output, loss

    def reset_parameters(self):
        return

    def clamp(self):
        return


class DropoutLayer(torch.nn.Module):
    def __init__(self, position, mode="Nodes", rem_prob=0.5, flatten_output=False):

        super(DropoutLayer, self).__init__()

        self.in_channels = None
        self.out_channels = None

        self.input_shape = None
        self.output_shape = None

        self.flatten_output = flatten_output
        self.mode = mode
        self.position = position      # number of dimensions except Batches and channels
        self.rem_prob = rem_prob
        self.spk_rec_hist = None
        self.training = True
        self.mask = None

        self.reset_parameters()
        self.clamp()

    def forward(self, x):
        batch_size = x.shape[0]
        if (self.position == "after_conv") :
            self.in_channels = x.shape[1]
            self.out_channels = x.shape[1]
            self.input_shape = (x.shape[3], x.shape[4])
            self.output_shape = (x.shape[3], x.shape[4])
            nb_steps = x.shape[2]
            mask=torch.ones_like(torch.mean(x,dim=2, keepdim=True))

        elif (self.position == "after_dense") :
            self.in_channels = 0
            self.out_channels = 0
            self.input_shape = (x.shape[2])
            self.output_shape = (x.shape[2])
            nb_steps = x.shape[1]
            mask = torch.ones_like(torch.mean(x, dim=1, keepdim=True))

        if self.mode == "Nodes":
            self.mask=torch.nn.functional.dropout(mask, p=self.rem_prob, training=self.training)
        elif self.mode == "Channels":
            self.mask=torch.nn.functional.dropout3d(mask, p=self.rem_prob, training=self.training)

        if self.training:
            x = x*self.mask

        # save spk_rec for plotting
        self.spk_rec_hist = x.detach().cpu().numpy()

        loss = 0 * (x).mean()

        if self.flatten_output:
            output = torch.transpose(x, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))
        else:
            output = x

        return output, loss

    def reset_parameters(self):
        self.mask = None

    def clamp(self):
        return


class BatchNormLayer(torch.nn.Module):
    def __init__(self, position, output_shape, out_channels, spike_params, train_params, eps=1e-8, flatten_output=False):

        super(BatchNormLayer, self).__init__()

        self.out_channels = out_channels
        self.output_shape = output_shape
        self.position = position
        self.eps = eps
        self.nb_steps = spike_params["nb_steps"]
        self.train_params = train_params
        self.intermediate_mode = train_params['BN_intermediate_mode']

        self.flatten_output = flatten_output
        self.spk_rec_hist = None
        self.training = True

        if (self.position == "input"):
            self.scale = torch.nn.Parameter(torch.empty(self.out_channels, *self.output_shape), requires_grad=train_params['BN_scale'])
            self.offset = torch.nn.Parameter(torch.empty(self.out_channels, *self.output_shape), requires_grad=train_params['BN_offset'])
        elif (self.position == "intermediate"):
            self.scale = torch.nn.Parameter(torch.empty(self.out_channels, 1, *self.output_shape), requires_grad=train_params['BN_scale'])
            if self.intermediate_mode=="with_offset":
                self.offset = torch.nn.Parameter(torch.empty(self.out_channels, 1, *self.output_shape), requires_grad=train_params['BN_offset'])
            elif self.intermediate_mode=="without_offset":
                self.offset = torch.nn.Parameter(torch.empty(self.out_channels, 1, *self.output_shape), requires_grad=False)

        self.reset_parameters()
        self.clamp()

    def forward(self, x):
        batch_size = x.shape[0]

        if (self.position == "input"):
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        elif (self.position == "intermediate"):
            x_mean = torch.mean(x, dim=(0,2), keepdim=True)
            x_var = torch.var(x, dim=(0,2), keepdim=True, unbiased=False)

        if (self.position == "input"):
            x_hat = (x - x_mean)/(torch.sqrt(x_var+self.eps))
        elif (self.position == "intermediate"):
            if self.intermediate_mode == "with_offset":
                x_hat = (x - x_mean)/(torch.sqrt(x_var+self.eps))
            elif self.intermediate_mode == "without_offset":
                x_hat = x/(torch.sqrt(x_var+self.eps))
        else :
            print("Invalid position value for BatchNormLayer!")
        x_hat = self.scale*x_hat + self.offset

        # save spk_rec for plotting
        self.spk_rec_hist = x_hat.detach().cpu().numpy()

        loss = 0 * (x_hat).mean()

        if self.flatten_output:
            output = torch.transpose(x_hat, 1, 2).contiguous()
            output = output.view(batch_size, self.nb_steps, self.out_channels * np.prod(self.output_shape))
        else:
            output = x_hat

        return output, loss

    def reset_parameters(self):
        torch.nn.init.constant_(self.scale, 1.)
        torch.nn.init.constant_(self.offset, 0.)

    def clamp(self):
        return


class ReadoutLayer(torch.nn.Module):

    "Fully connected readout"

    def __init__(self,  input_shape, output_shape, spike_fn, surrogate_sigma, w_init_mean, w_init_std, spike_params, train_params, eps=1e-8, time_reduction="mean", threshold=1):


        assert time_reduction in ["mean", "max", "mean_max", "latency", "max_latency"], 'time_reduction should be "mean" or "max" or "mean_max" or "latency" or "max_latency"'

        super(ReadoutLayer, self).__init__()


        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        self.threshold = threshold * np.ones(output_shape)
        self.threshold_param = torch.nn.Parameter(torch.empty(output_shape), requires_grad=False)
        self.latency_mode_inited = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.train_params = train_params
        self.time_step = spike_params["time_step"]
        self.tau_mem = 10e-3
        self.Leaky_Neuron = spike_params["Leaky_Neuron"]
        self.latency_output_mode=train_params['Readout_latency_output_mode']
        self.max_latency_output_mode=train_params['Readout_max_latency_output_mode']
        self.Readout_steps_start=train_params['Readout_steps_start']
        self.Readout_steps_end=train_params['Readout_steps_end']

        self.eps = eps
        self.time_reduction = time_reduction
        self.spike_fn = spike_fn

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=train_params['w'])
        if train_params['layer_beta_mode'] == 'one_per_neuron':
            self.beta = torch.nn.Parameter(torch.empty(output_shape), requires_grad=train_params['beta'])
        elif train_params['layer_beta_mode'] == 'one_per_layer':
            self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=train_params['beta'])
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=train_params['b'])
        self.b_latency = torch.nn.Parameter(torch.empty(output_shape), requires_grad=train_params['Readout_b_latency'])
        self.latency_scale = torch.nn.Parameter(train_params['Readout_latency_scale_val'] * torch.ones(1), requires_grad=train_params['Readout_latency_scale'])
        if train_params['surrogate_sigma_mode'] == 'all':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(self.output_shape), requires_grad=train_params['surrogate_sigma'])
        elif train_params['surrogate_sigma_mode']=='one':
            self.surrogate_sigma = torch.nn.Parameter(surrogate_sigma * torch.ones(1), requires_grad=train_params['surrogate_sigma'])

        self.reset_parameters()
        self.clamp()

        self.mem_rec_hist = None
        self.mthr_hist = None

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        h = torch.einsum("abc,cd->abd", x, self.w)

        with torch.autograd.profiler.record_function("label-sum"):
            norm = (self.w**2).sum(0)

        # membrane potential
        mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # memrane potential recording
        mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

        for t in range(nb_steps):
            # membrane potential update
            if (self.Leaky_Neuron):
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
            else:
                mem = mem * self.beta + h[:, t, :]
            mem_rec[:, t, :] = mem

        if self.time_reduction == "latency":
            mthr = torch.einsum("abc,c->abc", mem_rec, 1. / (norm + self.eps)) - self.b_latency
            spk_rec = self.spike_fn(mthr, self.surrogate_sigma)
            spk_rec_inverse = 1 - spk_rec
            spk_rec_inverse_cumprod = spk_rec_inverse.cumprod(dim=1)
            latency = spk_rec_inverse_cumprod.sum(dim=1, keepdim=False)
            if self.latency_output_mode=="method_1":
                output = self.latency_scale*(nb_steps-latency)
            elif self.latency_output_mode == "method_2":
                output = -1*torch.exp(self.latency_scale*latency)

        elif self.time_reduction == "max_latency":
            t = torch.range(start=1, end= nb_steps, step=1, dtype=x.dtype, device=x.device, requires_grad=False).unsqueeze(1).repeat((1,self.output_shape))
            w_sum = (self.w).sum(0)
            if self.max_latency_output_mode == "method_1":
                output = torch.max(mem_rec-self.b*w_sum*t/nb_steps, 1)[0]
            if self.max_latency_output_mode == "method_2":
                output = torch.mean(mem_rec/t, 1) / (norm + self.eps) - self.b

        elif self.time_reduction == "max":
            output = torch.max(mem_rec[:,self.Readout_steps_start:self.Readout_steps_end,:], 1)[0]/(norm+self.eps) - self.b

        elif self.time_reduction == "mean":
            output = torch.mean(mem_rec[:,self.Readout_steps_start:self.Readout_steps_end,:], 1) / (norm + self.eps) - self.b

        elif self.time_reduction == "mean_max":
            mem_rec_zero = torch.zeros_like(mem_rec)
            output = torch.mean(torch.max(mem_rec[:,self.Readout_steps_start:self.Readout_steps_end,:],mem_rec_zero[:,self.Readout_steps_start:self.Readout_steps_end,:]), 1) / (norm + self.eps) - self.b

        # save mem_rec for plotting
        if(self.time_reduction != "latency") :
            _, am = torch.max(output, 1)  # argmax over output units
            mthr = torch.einsum("abc,c->abc", mem_rec, 1. / (norm + self.eps))
            for i in range(batch_size):
                self.threshold[am[i]] = 0.998 * self.threshold[am[i]] + 0.002 * np.mean(mthr[i,:,am[i]].detach().cpu().numpy())
            self.threshold_param = torch.nn.Parameter(torch.tensor(self.threshold))
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        self.mthr_hist = mthr.detach().cpu().numpy()

        loss = None
        return output, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.input_shape)))

        if self.train_params['beta_init_method'] == "constant":
            # torch.nn.init.constant_(self.beta, float(np.exp(-self.time_step/self.tau_mem)))
            # torch.nn.init.constant_(self.beta, 1)
            torch.nn.init.constant_(self.beta, self.train_params['beta_constant_val'])
        elif self.train_params['beta_init_method'] == "normal":
            # torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=self.train_params['beta_normal_mean'],
                                  std=self.train_params['beta_normal_std'])
        elif self.train_params['beta_init_method'] == "uniform_":
            # torch.nn.init.uniform_(self.beta, a=0, b=1)
            torch.nn.init.uniform_(self.beta, a=self.train_params['beta_uniform_start'],
                                   b=self.train_params['beta_uniform_end'])

        if self.train_params['b_init_method'] == "constant":
            # torch.nn.init.constant_(self.b, 1.)
            torch.nn.init.constant_(self.b, self.train_params['b_constant_val'])
        elif self.train_params['b_init_method'] == "normal":
            # torch.nn.init.normal_(self.b, mean=1., std=0.01)
            torch.nn.init.normal_(self.b, mean=self.train_params['b_normal_mean'],
                                  std=self.train_params['b_normal_std'])
        elif self.train_params['b_init_method'] == "uniform_":
            torch.nn.init.uniform_(self.b, a=self.train_params['b_uniform_start'], b=self.train_params['b_uniform_end'])

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)

    def latency_mode_init(self, device, dtype):
        if self.latency_mode_inited.item() == 0 :
            print("latency_mode initialized!")
            self.time_reduction = "latency"

            state_dict = self.state_dict()
            # state_dict["b_latency"] = torch.from_numpy(self.threshold).to(device)
            state_dict["b_latency"] = self.threshold_param.detach()
            self.load_state_dict(state_dict)
            # with torch.no_grad():
            #     for name, param in self.named_parameters():
            #         if 'b_latency' in name:
            #             param.copy_(torch.from_numpy(self.threshold).to(device))

            self.latency_mode_inited = torch.nn.Parameter(torch.ones(1).to(device, dtype), requires_grad=False)
        else :
            print("latency_mode was previously initialized!")

    def print_params(self):
        print("Readout_time_reduction : ", self.time_reduction)
        print("Readout layer b_latency : ", self.b_latency)
        print("Readout layer latency_scale : ", self.latency_scale)
        print("Readout layer threshold_param : ", self.threshold_param)


class SurrogateHeaviside(torch.autograd.Function):

    # Activation function with surrogate gradient
    backward_mode = "sigmoid"
    surrogate_mode = "heaviside"

    @staticmethod
    def forward(ctx, input, sigma):
        if SurrogateHeaviside.surrogate_mode == "heaviside":
            output = torch.zeros_like(input)
            thr = 0
            output[input > thr] = 1.0
        elif SurrogateHeaviside.surrogate_mode == "non_heaviside":
            if SurrogateHeaviside.backward_mode == "sigmoid":
                output = torch.sigmoid(sigma*input)
            elif SurrogateHeaviside.backward_mode == "rectangle":
                output = torch.zeros_like(input)
                output[input > 0] = 1.0
            elif SurrogateHeaviside.backward_mode == "fast_sigmoid_abs":
                output = 0.5 * (1 + (sigma * input)/(1 + torch.abs(sigma * input)))
            elif SurrogateHeaviside.backward_mode == "fast_sigmoid_tanh":
                output = 0.5 * (1 + torch.tanh(sigma*input))
        else :
            print("Error : mode not exist for surrogate!")
        ctx.save_for_backward(input, sigma)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, sigma = ctx.saved_tensors
        grad_input = grad_output.clone()

        # approximation of the gradient using sigmoid function
        if SurrogateHeaviside.backward_mode=="sigmoid":
            grad = grad_input*sigma*torch.sigmoid(sigma*input)*torch.sigmoid(-sigma*input)
            grad_sigma = grad_input * input * torch.sigmoid(sigma * input) * torch.sigmoid(-sigma * input)

        elif SurrogateHeaviside.backward_mode=="rectangle":
            grad = grad_input*(torch.abs(input)<(1/sigma))
            grad_sigma = torch.zeros_like(grad_input)

        elif SurrogateHeaviside.backward_mode=="fast_sigmoid_abs":
            grad = grad_input * (0.5*sigma)/(1+torch.abs(sigma*input))**2
            grad_sigma = grad_input * (0.5*input)/(1+torch.abs(sigma*input))**2

        elif SurrogateHeaviside.backward_mode == "fast_sigmoid_tanh":
            grad = grad_input * (0.5*sigma)*(1-(torch.tanh(sigma*input))**2)
            grad_sigma = grad_input * (0.5*input)*(1-(torch.tanh(sigma*input))**2)

        return grad, grad_sigma

