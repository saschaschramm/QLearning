


grid = [
    ["S", "F", "F", "F"],
    ["F", "H", "F", "H"],
    ["F", "F", "F", "H"],
    ["H", "F", "F", "G"]
]





import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 4, figsize=(4, 4))
fig.subplots_adjust(hspace=0, wspace=0)

foo=plt.gca()
foo.set_ylim(foo.get_ylim()[::-1])

for y in range(4):
    for x in range(4):
        ax[y, x].xaxis.set_major_locator(plt.NullLocator())
        ax[y, x].yaxis.set_major_locator(plt.NullLocator())

        cell = grid[y][x]

        if cell == "F":
            ax[y, x].set_facecolor('xkcd:very light green')
        elif cell == "H":
            ax[y, x].set_facecolor('xkcd:light pink')
            

        plt.text(x + 0.5-3, y + 0.5-3, cell,
                 horizontalalignment='center',
                 verticalalignment='center',

                 )

fig.savefig('grid2.png', bbox_inches='tight')


plt.show()