from csv import reader
def int_or_float(x):
    try:
        return int(x)
    except:
        return float(x)

# read the csv, export data = [ ...[rowdata]]
# mode = 'str', 'int', 'float'
def read_file(path, mode='str'):
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        list_of_rows = list(csv_reader)
        if mode == 'float':
            list_of_rows = [ [int_or_float(j) for j in i] for i in list_of_rows]
        if mode == 'int':
            list_of_rows = [ [int(j) for j in i] for i in list_of_rows]
        return list_of_rows

# write the csv, data = [ ...[rowdata]]
def write_file(path, arr = [['this', 'is'], ['an', 'example']]):
    g = open(path, "w")
    for i in arr:
        g.write(",".join(str(j) for j in i) + "\n")
    g.close()
