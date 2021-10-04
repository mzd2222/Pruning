from visdom import Visdom

vis = Visdom()



file_path = "./none/acc.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

vis.line([[0. for _ in range(len(lines))]], [0.], win='train_acc', env='main',
         opts=dict(title='train acc', legend=[str(i) for i in range(len(lines))], xlabel="EPOCH", ylabel="ACC"))

acc = []

for _ in range(len(lines)):
    acc.append([])

for i in range(len(lines)):
    a = lines[i].split(" ")[:-1]
    for j in range(len(a)):
        a[j] = float(a[j])

    acc[i] = a
# print(acc)
for i in range(len(acc[0])):
    # print(acc[0][i], acc[1][i], acc[2][i])
    vis.line([[acc[k][i] for k in range(len(lines))]], [i], win='train_acc', update='append')
