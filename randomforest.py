from random import seed, randrange
from csv import reader
from math import sqrt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# Load a CSV File
def load_file(filename):
    data = list() #make an empty list
    with open(filename, 'r') as file:
        csv_reader = reader(file) #read the csv file
        for row in csv_reader:
            if not row: continue
            data.append(row)
     # delete the last column
    	for i in range(1,len(data)):
		#del data[i][0] 
		del data[i][0]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		del data[i][2]
		data[i-1] = [float(x) for x in data[i]]
		
	data[len(data)-1] = [float(x) for x in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]	
	return data

# Calculate accuracy 
def accuracy_metric(actual, predicted):
    a = 0
    for i in xrange(len(actual)):
        if actual[i] == predicted[i]: a += 1
    return 100. * a / float( len(actual) )

# The entry point of the applicatiom
# It evalutes the random forest using k runs
def random_forest_algorithm(data, max_tree_depth, min_size, sample_size, trees, features, num_runs):
    #Split data
    run_size = len(data) / num_runs
    runs = list()
    copy = list(data)
    acc = list()
    prec = list()
    rec = list()
    fs = list()
    sup = list()
    #Randomly generate runs
    for i in xrange(num_runs):
        run = list()
        while len(run) < run_size:
            index = randrange(len(copy))
            run.append(copy.pop(index))
        runs.append(run)
    
        
    # For every run build a tree    
    for run in runs:
        train = list(runs)
        train.remove(run)
        train = sum(train, [])
        test = list()
        for row in run:
            r_copy = list(row)
            test.append(r_copy)
            r_copy[-1] = None
            
        new_tree_list = list()
        for i in range(trees):            
            #  With replacement create a random subsample from data
            sample = list()
            total_sample = round(len(train) * sample_size)
            while len(sample) < total_sample:
                index = randrange(len(train))
                sample.append(train[index])            
            #Build the tree using the best split for the data
            new_tree = find_split(sample, features)
            
            split_node(new_tree, max_tree_depth, min_size, features, 1)            
            
            new_tree_list.append(new_tree)
            
        predict = [tree_predict(new_tree_list, row) for row in test]            
        #predicted = random_forest(train, test, features, max_tree_depth, min_size, sample_size, trees)
        actual = [row[-1] for row in run]
        accuracy = accuracy_metric(actual, predict)
        #print confusion_matrix(actual, predict)
        precision, recall, fscore, support = score(actual, predict)
        #print "precision"        
        #print precision
        #print "recall"
        #print recall
        #print "fscore"
        #print fscore
        #print "support"
        #print support
        prec.append(precision[0])
        rec.append(recall[0])
        fs.append(fscore[0])
        sup.append(support[0])
        acc.append(accuracy)
    return prec,rec,fs,sup,acc

# Select the best split point for a data
def find_split(data, total_features):    
    global_groups, global_index, global_value, global_score = None, 999, 999, 999
    classval = list(set(row[-1] for row in data))
    feature = list()
    while len(feature) < total_features:
        i = randrange(len(data[0])-1)
        if i not in feature: feature.append(i)
    for i in feature:
        for row in data:
            #Split the data based on a feature            
            left, right = list(), list()
            for d in data:
                if d[i] >= row[i]: right.append(row)
                else: left.append(row)
            grp = left, right            
                        
            #Calculate the gini index
            gini = 0.0
            for one_class in classval:
                for group in grp:
                    size = len(group)
                    if size != 0:
                        prop = [row[-1] for row in group].count(one_class) / float(size)
                        gini += ((1.0 - prop) * prop)            
            
            if global_score > gini: 
                global_score, global_groups, global_index, global_value = gini, grp, i, row[i]
    return {'pointer': global_index, 'val': global_value, 'node': global_groups}

# Create a leaf node
def leaf_node(group):
    output = [row[-1] for row in group]
    return max(set(output), key=output.count)

# Recursively split node or make it a leaf node
def split_node(node, max_depth, min_size, n_features, depth):
    left, right = node['node']
    del(node['node'])
    
    if not left or not right:
        node['left'] = node['right'] = leaf_node(left + right)
        return
    
    if max_depth <= depth:
        node['left'], node['right'] = leaf_node(left), leaf_node(right)
        return
    
    if min_size >= len(left):
        node['left'] = leaf_node(left)
    else:
        node['left'] = find_split(left, n_features)
        split_node(node['left'], max_depth, min_size, n_features, depth+1)
    
    if min_size >= len(right):
        node['right'] = leaf_node(right)
    else:
        node['right'] = find_split(right, n_features)
        split_node(node['right'], max_depth, min_size, n_features, depth+1)
 
 
def tree_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Prediction for a decision tree
def predict(node, row):
    if row[node['pointer']] < node['val']:
        if isinstance(node['left'], dict): return predict(node['left'], row)
        else: return node['left']
    else:
        if isinstance(node['right'], dict): return predict(node['right'], row)
        else: return node['right']



# Main Program or the entry point of our program
seed(10)
#Load Data
data = load_file('train.csv')

min_size = 1
sample_size = 1.0
folds = 5
max_tree_depth = 10

features = int(sqrt(len(data[0])-1))
for trees in [1, 5]:
    prec,rec,fs,sup,acc = random_forest_algorithm(data, max_tree_depth, min_size, sample_size, trees, features, folds)
    print('Random Forest Trees: %d' % trees)
    print('Precision: %s' % prec)
    print('Recall: %s' % rec)
    print('fscore: %s' % fs)
    print('Support: %s' % sup)
    print('Accuracy: %s' % acc)
    
    print('Average Accuracy: %.3f%%' % (sum(acc)/float(len(acc))))
    print('')