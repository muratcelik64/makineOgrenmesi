# sembol_list
# author: muratcelik64
# veri seti içerisinde bulunabilecek sembolleri içerir

import pandas as pd

symbols2 = ['☺☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲▼"#$%&()*+,-./:;<=>?@']

symbols = ['☺','☻','♥','♦','♣','♠','•','◘','○','◙',
           '♂','♀','♪','♫','☼','►','◄','↕','‼','¶',
           '§','▬','↨','↑','↓','→','←','∟','↔','▲',
           '▼',' ','!','"','#','$','%','&','(',
           ')','*','+',',','-','.','/',':',';','<',
           '=','>','?','@']

def symbol_to_character(dataCol):
    for ix in dataCol:
        for y in range(len(symbols)):
            if(ix.count(symbols[y])>=1):
                dataCol = dataCol.str.replace(symbols[y], "_")
    return dataCol