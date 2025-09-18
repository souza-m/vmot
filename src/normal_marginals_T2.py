# -*- coding: utf-8 -*-
"""
Created on Jul 18 2025
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
Example with cross-product cost and normal marginals, d = 2, T = 2
"""c vbnm
if __name__ == "__main__":
    I = 100
    n_points = 1000000
    timers = []
    for monotone in [True, False]:
        existing_i = 0
        mono_label = 'mono' if monotone else 'full'
        print(mono_label)
        t0 = time.time()
        timers.append(t0)
        if existing_i == 0:
            print('\niteration 1 (new model)\n')
            u, v, x, y = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y)
            ws = vmot.generate_working_sample_T2(u, v, x, y, c)
            print('sample generated, shape ', ws.shape)
            model, opt = vmot.generate_model(d, T, monotone = monotone)
            print('model generated')
            D_series, H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i = 1
            vmot.dump_results([model, opt, D_series], f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
        while existing_i < I:
            print(f'\niteration {existing_i+1}\n')
            u, v, x, y = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y)
            ws = vmot.generate_working_sample_T2(u, v, x, y, c)
            print('sample generated, shape ', ws.shape)
            print(f'\nloading model {existing_i}')
            model, opt, D_series = vmot.load_results(f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
            _D_series, _H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i += 1
            print('models updated')
            D_series = D_series + _D_series
            H_series = H_series + _H_series
            vmot.dump_results([model, opt, D_series], f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
        t1 = time.time()
        timers.append(t1)
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))