def calculWeight(input, sizeH, output):
    precedent = input
    sizeH.append(output)
    sum = 0
    for enCours in sizeH:
        sum += precedent * enCours
        precedent = enCours

    return sum

def calculPerceptrons(input, sizeH, output):
    sizeH.append(output)
    sum = 0
    for enCours in sizeH:
        sum += enCours

    return sum
