import csv
import random
import matplotlib.pyplot as plt

write_file = open('data_file.csv', 'w')
csv_writer = csv.writer(write_file)

list1 = [3]*50
list2 = []
for i in range(0,50):
    list2.append(random.randint(0,7))

for i in range(0,50):
    csv_writer.writerow([list1[i], list2[i]])

write_file.close()
    
read_file = open('data_file.csv')
csv_reader = csv.reader(read_file)
for r in csv_reader:
    print(r)

read_list1 = []
read_list2 = []
for row in csv_reader:
    read_list1.append(int(row[0]))
    read_list2.append(int(row[1]))

read_file.close()

combined_list = read_list1 + read_list2

only_figure, (left_subplot, right_subplot) = plt.subplots(1, 2, figsize=(8, 4))
left_subplot.plot(combined_list, linestyle='--')

list_x_values = []
for i in range(0,50):
    list_x_values.append(i)

right_subplot.scatter(list_x_values, list1,color='black', marker = '^')
right_subplot.scatter(list_x_values, list2,color='red', marker = '*')
plt.show()
