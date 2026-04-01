arr = [4,5,0,1,9,0,5,0]

number_array = []
zero_count = 0
for i in arr:
    if i == 0:
        zero_count += 1
    else:
        number_array.append(i)

number_array.extend([0] * zero_count)

print(number_array)