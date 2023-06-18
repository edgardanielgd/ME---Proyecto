
def compareArrays( arr1 , arr2 ):
    # Order is relevant and its not the same to have the sequence A B than B A
    if len(arr1) != len(arr2):
        return False
    
    for i in range( len(arr1) ):
        if arr1[i] != arr2[i]:
            return False

    return True