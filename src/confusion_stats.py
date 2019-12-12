def extract_nums(line):
	line = line.strip()
	line = line.replace('[', '').replace(']', '')
	line = line.split(' ')
	line = [int(x) for x in line if x != '']
	return line

def correct_class_ratio(nums):
	good = 0
	bad = 0
	for x in range(len(nums)):
		for y in range(len(nums)):
			if x == y:
				good += nums[x][y]
			else:
				bad += nums[x][y]
	return good / (good + bad)

def any_class_ratio(nums):
	good = 0
	bad = 0
	for x in range(len(nums)):
		for y in range(len(nums)):
			if (x == 0) == (y == 0):
				good += nums[x][y]
			else:
				bad += nums[x][y]
	return good / (good + bad)

with open('results/mean_variance_softmax_regression/test_acc.txt') as f:
	with open('results/mean_variance_softmax_regression/test_confusion.txt', 'w') as o:
		line_buffer = []
		for line in f:
			line_buffer.append(line)
			if len(line_buffer) == 4:
				nums = [extract_nums(x) for x in line_buffer]
				o.write(str(nums) + " " + str(round(correct_class_ratio(nums), 3)) + " " + str(round(any_class_ratio(nums), 3)) + "\n")
				line_buffer = []