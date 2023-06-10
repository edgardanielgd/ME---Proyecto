from nltk.tree.tree import Tree

# This class aims to parse a given tree to Chomsky Normal Form

# One single method is implemented in order to preserve structure
# Taken from: https://www.nltk.org/_modules/nltk/tree/transforms.html#chomsky_normal_form
def ParseChomskyNormalForm( original_parse_tree ):
    # Lets parse this probabilistic grammar to Chomsky Normal Form
    nodeList = [(original_parse_tree, [original_parse_tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):

            # parent annotation
            parentString = ""
            originalNode = node.label()

            # add children to the agenda before we mess with them
            for child in node:
                nodeList.append((child, parent))

            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []  # delete the children

                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1, numChildren - 1):

                    newHead = "{}{}<{}>{}".format(
                        originalNode,
                        "|",
                        "-".join(
                            childNodes[i : min([i + 999, numChildren])]
                        ),
                        parentString,
                    )  # create new head
                    newNode = Tree(newHead, [])
                    curNode[0:] = [nodeCopy.pop(0), newNode]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]
