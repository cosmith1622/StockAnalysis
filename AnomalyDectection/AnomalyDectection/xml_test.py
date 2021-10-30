import os 
def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
          
    return list
      

xml = None
item_data_1 ={'Confiedential=':'N','estimated=':'N', 'mrdc=':'II', 'value':'test'}
item_data_2 ={'Confiedential=':'N','estimated=':'N', 'mrdc=':'II', 'value':'test1'}
header = '<?xml version="1.0" encoding="UTF-8"?>'
namespace = '<report series="IICW">'
newline = '\n'
tab = '\t'
left_item = "<itemData>"
right_charat = ">"
right_item = "</itemData>"
left_value = "<value>"
right_value = "</value>"
item_list = []
item_list.append(item_data_1)
item_list.append(item_data_2)
list_of_keys = getList(item_list[0])
xml = header + newline + tab + namespace
save_path_file = r'c:\users\cosmi\onedrive\desktop\xml_test.xml'
for x in item_list:

    xml = xml + newline 
    item_node = tab + tab + left_item +  list_of_keys[0] + '"' + x[list_of_keys[0]] + '"' + " " +  list_of_keys[1] + '"' + x[list_of_keys[1]] + '"' + " " + list_of_keys[2] +  '"' + x[list_of_keys[2]] + '"' + right_charat
    xml = xml + item_node
    xml = xml + newline
    value_node = tab + tab + tab + left_value + x[list_of_keys[3]]  + right_value
    xml = xml + value_node
    xml = xml + newline 
    xml = xml + tab + tab + right_item
print(xml)
print(xml)
with open(save_path_file, "w") as f:
    f.write(xml) 
    
    
    
             
             
