import os

class lable_map_creator():
    def create_lable_map(enteries, saveloc):
        if len(enteries) == 0:
            return
        path = os.path.join(saveloc,'label_map.pbtxt' )
        with open(path, 'a') as lables:
            count = 1
            for i in enteries:
                i = i.strip()
                lables.write('item{\n')
                lables.write('\tid :{}'.format(int(count)))
                lables.write('\n')
                lables.write("\tname :'{0}'".format(str(i)))
                lables.write('\n')
                lables.write('}\n')
                count = count + 1