def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie_ids.txt and returns a
    #list of the titles
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie_ids.txt
    #   and returns a list of the titles in movieList.

    movieList = []

    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            movieName = line.split(' ', 1)[1].strip()
            movieList.append(movieName)

    return movieList
