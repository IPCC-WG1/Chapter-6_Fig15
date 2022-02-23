import os

def make_folders(path):
    """
    Takes path and creates to folders
    :param path: Path you want to create (if not already existant)
    :return: nothing

    Example (want to make folders for placing file myfile.png:
    >>> path='my/folders/myfile.png'
    >>> make_folders(path)
    """
    path = extract_path_from_filepath(path)
    split_path = path.split('/')
    if (path[0] == '/'):

        path_inc = '/'
    else:
        path_inc = ''
    for ii in range(0,len(split_path)):
        # if ii==0: path_inc=path_inc+split_path[ii]
        path_inc = path_inc + split_path[ii]
        if not os.path.exists(path_inc):
            os.makedirs(path_inc)
        path_inc = path_inc + '/'

    return

def extract_path_from_filepath(file_path):
    """
    ex: 'folder/to/file.txt' returns 'folder/to/'
    :param file_path:
    :return:
    """

    st_ind=file_path.rfind('/')
    foldern = file_path[0:st_ind]+'/'
    return foldern


def set_fontsize(ax, SM=8, MED=10, BIG=12):
    # ax.title.set_fontsize(SM)
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(SM)
    ax.title.set_fontsize(SM)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(SM)



def broken_axis(axs, break_between, ax_break='y'):
    """
    Plot the same plot in axis and break the axis between limits
    given in break_between
    :param axs: list of axis
    :param break_between: list [low_lim, high_lim]
    :param ax_break: 'x' or 'y'
    :param d: size of break symb
    :return:
    Example:
    >>> fig, axs = plt.subplots(3,3)
    >>> broken_axis(axs[0,1:], [3,10], ax_break='x')
    >>> plt.show()
    """
    spines = {'y': ['bottom', 'top'], 'x': ['right', 'left']}
    # remove spines between plots
    for s, ax in zip(spines[ax_break], axs):
        ax.spines[s].set_visible(False)

    if ax_break == 'y':
        # remove xaxis in between
        axs[0].xaxis.set_visible(False)
        # add dash on axis:
        _add_break_line(axs[0], where='bottom')
        _add_break_line(axs[1], where='top')
        # set limits on the plot:
        lims = axs[1].get_ylim()
        axs[1].set_ylim([lims[0], break_between[0]])
        lims = axs[0].get_ylim()
        axs[0].set_ylim([break_between[1], lims[1]])
    elif ax_break == 'x':
        # Set ticks on left.
        axs[0].yaxis.tick_left()
        # remove ticks to right
        axs[0].tick_params(labelright=False)  # 'off')
        axs[1].yaxis.tick_right()
        # add dash on axis:
        _add_break_line(axs[0], where='right', axis='x')
        _add_break_line(axs[1], where='left', axis= 'x')

        lims = axs[0].get_xlim()
        axs[0].set_xlim([lims[0], break_between[0]])
        lims = axs[1].get_xlim()
        axs[1].set_xlim([break_between[1], lims[1]])
def _add_break_line(ax,where='top', fs= 16, wgt='bold', axis='y'):
    """
    Adds break line to plot
    :param ax: the axis to add to
    :param where: 'top','bottom','left', 'right' (where to add)
    :param fs: size of the line (font size)
    :param wgt: weight of the font
    :param axis: which axis
    :return:
    """
    anno_opts = dict( xycoords='axes fraction',
                      va='center', ha='center', rotation=-45, fontsize=fs, fontweight=wgt)
    if where in ['top', 'right']:k=1
    else: k=0
    if axis=='y':
        ax.annotate('|',xy=(0,k ), **anno_opts)
        ax.annotate('|',xy=(1,k ), **anno_opts)
    else:
        ax.annotate('|',xy=(k,0), **anno_opts)
        ax.annotate('|',xy=(k,1 ), **anno_opts)
