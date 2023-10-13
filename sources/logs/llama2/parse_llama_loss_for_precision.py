import matplotlib.pyplot as plt
import numpy as np
import os


def parse_log_file(log_file, tag="samples per second:"):
    total_time = 0.0
    steps = 0
    train_losses = {}
    train_perf = {}
    iterations = []
    lrs = []
    grad_norm = []

    with open(log_file, encoding='gb18030', errors="ignore") as fr:
        for line in fr:
            if 'iteration ' in line and "number of skipped iterations" in line and int(
                    line.split("number of skipped iterations:")[1].split(" | ")[0]) > 0:
                print(line)
            elif ('iteration ' in line and "elapsed time per iteration (ms): " in line and "lm loss:" in line):
                step_time = float(line.split("elapsed time per iteration (ms):")[1].split(" | ")[0]) / 1000.0
                loss = float(line.split("lm loss:")[1].split(" | ")[0])
                iteration = int(line.split("iteration ")[1].split("/")[0].strip())

                train_losses[iteration] = loss
                train_perf[iteration] = step_time
                steps += 1
                # assert iteration == steps, "{} vs {}".format(steps, line)
                lrs.append(float(line.split("learning rate: ")[1].split(" | ")[0]))
                # grad_norm.append(float(line.split("grad norm: ")[1].split(" | ")[0]))
                grad_norm.append(0.0)

                # train_fw.write('\n' + str(loss))
            elif 'iteration' in line and "samples per second:" in line:
                train_losses.append('nan')
                iteration = float(line.split("iteration")[1].split("/")[0])
                iterations.append(iteration)
            # elif "Training time" in line:
            #     items = line.split("Training time")
            #     h, m, s = [float(i) for i in items[1].split(":")]
            #     total_time = h * 3600 + m * 60 + s

    print("parsed {}".format(log_file))

    return train_losses, train_perf, iterations, lrs, grad_norm

def loss_compare_plot(files, freq=50, labels="gpu, npu"):
    plt.figure(figsize=(10, 10), dpi=80)
    plots = 7
    plt.subplot(plots, 1, 1)
    baseline, baseline_perf, baseline_steps, baseline_lrs, baseselin_grad_norm = parse_log_file(files[0])
    data, perf, steps, lrs, grad_norm = parse_log_file(files[1])
    ae = {}
    ape = {}
    baseline_total_time = 0.0
    total_time = 0.0
    steps = 0
    for step in baseline:
        if step in data:
            ae[step] = abs(data[step] - baseline[step])
            ape[step] = abs(data[step] - baseline[step]) / (baseline[step] + 1e-10)
        if step in baseline_perf and step in perf:
            steps += 1
            baseline_total_time += baseline_perf[step]
            total_time += perf[step]
    baseline_avg_TPS = 1.0 / (baseline_total_time / steps / 32) * 4096 / 8
    avg_TPS = 1.0 / (total_time / steps / 32) * 4096 / 8
    speedup = avg_TPS/baseline_avg_TPS

    mean_ae, max_ae = np.array(list(ae.values())).mean(), np.array(list(ae.values())).max()
    mean_ape, max_ape = np.array(list(ape.values())).mean(), np.array(list(ape.values())).max()

    # steps = min(len(baseline), len(data))
    # baseline = baseline[0:steps:freq]
    # data = data[0:steps:freq]
    # ae = ae[0:steps:freq]
    # ape = ape[0:steps:freq]

    plt.plot(np.array(list(baseline.keys())), np.array(list(baseline.values())), label=labels.split(",")[0].strip())
    plt.plot(np.array(list(data.keys())), np.array(list(data.values())), label=labels.split(",")[1].strip())

    plt.title("Performance(tokens/p/s): {:.4f} vs {:.4f} {:.2f}X \n  {} \n vs {} \n convergence: {} \n Loss Comparison".format(
        avg_TPS, baseline_avg_TPS, speedup, files[0], files[1], convergence(list(ae.values()), len(ae))))
    plt.legend()

    plt.subplot(plots, 1, 2)
    plt.plot(np.array(list(ape.keys())), np.array(list(ape.values())))
    plt.title("Relative Loss Error MAX: {:.4f}, MEAN: {:.4f}".format(max_ape, mean_ape))

    plt.subplot(plots, 1, 3)
    plt.plot(np.array(list(ae.keys())), np.array(list(ae.values())))
    plt.title("Absolute Loss Error MAX: {:.4f}, MEAN: {:.4f}".format(max_ae, mean_ae))

    plt.subplot(plots, 1, 4)
    plt.plot(baseline_lrs)
    plt.plot(lrs)
    plt.title("Learning Rate")

    plt.subplot(plots, 1, 5)
    plt.plot(baseselin_grad_norm)
    plt.plot(grad_norm)
    plt.title("Grad Norm")

    plt.subplot(plots, 1, 6)
    plt.plot([abs(grad_norm[i] - baseselin_grad_norm[i]) for i in range(min(len(baseselin_grad_norm), len(grad_norm)))])
    plt.title("Grad Norm Absolute Error")

    plt.subplot(plots, 1, 7)
    plt.plot([abs(lrs[i] - baseline_lrs[i]) for i in range(min(len(lrs), len(baseline_lrs)))])
    plt.title("Learning Rate Absolute Error")


    plt.savefig(os.path.join("../../images/llama2", "llama2_7b_shape_fp16_layer32_{}.png".format("finetune" if "finetune" in labels else "pretrain")))
    plt.show()
def convergence(loss_metrics, static_num):
    tp9_idx = int(static_num * 0.9)
    tp99_idx = int(static_num * 0.99)
    tp999_idx = int(static_num * 0.999)
    tp9999_idx = int(static_num * 0.9999)
    sorted_static_metrics = sorted(loss_metrics)
    print(
        f'最大偏差： {max(loss_metrics)}, d_tp999 = {sorted_static_metrics[tp9999_idx]}, d_tp999 = {sorted_static_metrics[tp999_idx]}, d_tp99 = {sorted_static_metrics[tp99_idx]}')
    # print(mses)
    print('绝对偏差平均值：', round(abs(sum(loss_metrics) / static_num), 4))

    return f'\nloss_err_mean: {round(abs(sum(loss_metrics) / static_num), 4)}' \
           f'\nloss_err_tp99: {round(sorted_static_metrics[tp99_idx], 4)}' \
           f'\nloss_err_tp999: {round(sorted_static_metrics[tp999_idx], 4)}' \
           f'\nloss_err_tp9999: {round(sorted_static_metrics[tp9999_idx], 4)}'

files = ["GPU_llama2_7b_shape_fp16_layer32_8p_pretrain.out",
         "NPU_llama2_7b_shape_fp16_layer32_8p_pretrain.out"]
loss_compare_plot(files=files, freq=1, labels="gpu_pretrain, npu_pretrain")

files = ["GPU_llama2_7b_shape_fp16_layer32_8p_finetune.out",
         "NPU_llama2_7b_shape_fp16_layer32_8p_finetune.out"]
loss_compare_plot(files=files, freq=1, labels="gpu_finetune, npu_finetune")