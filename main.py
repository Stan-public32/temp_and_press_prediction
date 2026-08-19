import numpy as np

temps = []
press = []

with open('data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        result = line.split('\t')
        if len(result) == 4:
            temps.append(float(result[2].replace(',','.')))
            press.append(float(result[3].replace(',','.')))
temps = np.array(temps)
press = np.array(press)

temp_mean, temp_std = temps.mean(), temps.std()
press_mean, press_std = press.mean(), press.std()
temps_norm = (temps - temp_mean) / temp_std
press_norm = (press - press_mean) / press_std

window = 8
n_sets = len(temps) - window - 1
max_cycles = 2000
alpha = 0.0023
weights = np.ones((window*2, 2))
weights /= 100

X = []
Y = []
preds = []
corr = []

for i in range(n_sets):
    temp_X = []
    for j in range(window):
        temp_X.append(temps_norm[j+i])
        temp_X.append(press_norm[j+i])
    X.append(np.array(temp_X))
    Y.append(np.array(((temps_norm[i+window]),(press_norm[i+window]))))
X = np.array(X)
Y = np.array(Y)

for cycles in range(max_cycles):
    preds = X @ weights
    deltas = preds - Y
    error = np.mean(deltas ** 2)
    corr = ((X.T) @ deltas) * alpha
    weights -= corr
    if (cycles % 200) == 0:
        print("Cycle: " + str(cycles) + " ; Err: " + str(error))

with open('wgts.txt', 'w', encoding='utf-8') as f:
    for out in weights:
        f.write(str(out[0]) + "\t" + str(out[1]) + "\n")

input = []
print("Температуры и давления для предсказания:")
for i in range(window):
    print(str(temps[i-window]) + " ;  " + str(press[i-window]))
    input.append(temps_norm[i-window])
    input.append(press_norm[i-window])
input = np.array(input)
output = input @ weights
temp_pred = (output[0] * temp_std) + temp_mean
press_pred = (output[1] * press_std) + press_mean

print("Предсказанная температура: " + format(temp_pred,'.2f') + "\nПредсказанное давление: " + format(press_pred,'.1f'))
