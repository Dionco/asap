import numpy as np
from numba import jit

@jit(nopython=True)
def sign(a):
    '''Returns the sign of float'''
    if a < 0:
        sign = -1
    elif a > 0:
        sign = +1
    elif a == 0:
        sign = 0
    return sign

@jit(nopython=True)
def select_line(ref, wvl, flux):
    ## Line selection
    lim = 400 # km/s  We define a limit to not exceed
    d = (ref - wvl)**2
    i = np.where(d==np.min(d))[0][0] # position of closest point to line
    deriv = np.diff(flux) # Flux derivative

    ## Now we seach for when the derivative changes sign
    j = 1
    shift = (wvl[i+j]-wvl[i])/wvl[i]*(3*1e5) # We keep track of the shift
    deriv_sign = sign(deriv[i+j+1])
    while shift<lim: # Do not go too far
        if sign(deriv[i+j]) != deriv_sign:
            lb = shift
            break
        else:
            shift = (wvl[i+j]-wvl[i])/wvl[i]*(3*1e5)
            lb = shift
        j+=1
    j = -1
    shift = (wvl[i]-wvl[i+j])/wvl[i]*(3*1e5)
    deriv_sign = sign(deriv[i+j-1])
    while shift<lim:
        if sign(deriv[i+j]) != deriv_sign:
            hb = shift
            break
        else:
            shift = (wvl[i]-wvl[i+j])/wvl[i]*(3*1e5)
        j-=1
    hws = (hb+lb)/2
    return hws

@jit(nopython=True)
def select_lines(lines, wvl, flux):
    hwss = np.empty(lines.shape)
    for i in range(len(lines)):
        if (lines[i]<wvl[0]) | (lines[i]>wvl[-1]):
            hws = 0
        else:
            hws = select_line(lines[i], wvl, flux)
        hwss[i] = hws
    return hwss

def gen_bounds(wlines, hwss):
    bounds = []
    for i in range(len(wlines)):
        lim = hwss[i]*wlines[i]/(3*1e5)
        bounds.append([wlines[i]-lim, wlines[i]+lim])
    return bounds

def filter(wlines, bounds):
    nbounds = []
    nwlines = []
    idxtoremove = []
    for i in range(len(bounds)):
        if (bounds[i][1]-bounds[i][0])==0:
            idxtoremove.append(i)
            continue
        else:
            nbounds.append(bounds[i])
            nwlines.append(wlines[i])
    return nwlines, nbounds, idxtoremove

def increase_width(bounds, pad=0.05):
    '''Increase the width of bounds by a padding factor
    
    Performance optimization: vectorized operation using numpy
    '''
    if bounds is None or (hasattr(bounds, '__len__') and len(bounds) == 0):
        return []
    
    bounds_arr = np.array(bounds)
    
    # Handle empty array case
    if bounds_arr.size == 0:
        return []
    
    widths = bounds_arr[:, 1] - bounds_arr[:, 0]
    
    # Vectorized padding calculation
    pad_amount = pad * widths
    nbounds_arr = np.column_stack([
        bounds_arr[:, 0] - pad_amount,
        bounds_arr[:, 1] + pad_amount
    ])
    
    return nbounds_arr.tolist()

def merge(bounds):
    '''Merge regions that are overlapping
    
    Performance optimizations:
    - Handle edge cases early
    - Use numpy array operations for comparisons
    - Simplify logic flow
    '''
    if bounds is None or (hasattr(bounds, '__len__') and len(bounds) == 0):
        return []
    
    if len(bounds) == 1:
        return bounds if isinstance(bounds, list) else bounds.tolist()
    
    # Convert to numpy array for easier manipulation
    bounds_arr = np.array(bounds)
    
    # Handle empty array case
    if bounds_arr.size == 0:
        return []
    
    # Filter empty bounds upfront
    non_empty = bounds_arr[:, 1] - bounds_arr[:, 0] > 0
    if not np.any(non_empty):
        return []
    
    bounds_arr = bounds_arr[non_empty]
    
    if len(bounds_arr) == 1:
        return bounds_arr.tolist()
    
    nbounds = []
    i = 0
    
    while i < len(bounds_arr):
        current_bound = bounds_arr[i].copy()
        j = i + 1
        
        # Merge all overlapping bounds
        while j < len(bounds_arr) and bounds_arr[j][0] < current_bound[1]:
            current_bound[1] = max(current_bound[1], bounds_arr[j][1])
            current_bound[0] = min(current_bound[0], bounds_arr[j][0])
            j += 1
        
        nbounds.append(current_bound.tolist())
        i = j
    
    return nbounds

def make_regions(wvl, flux, bounds, mask, length=400):
    '''
    Make regions from all the information on the lines and regions
    '''
    edges = int(1/8*length)
    length -= 2*edges
    # currentorder = orders[0]
    # order = orders[0]
    # while order==currentorder:
    lb, hb = bounds[0]
    if wvl[0]>=lb: # If the bounds were wider than the region
        start=0
    else:
        start = np.where(wvl[wvl<lb])[0][-1] # Start as close as possible to bound
    end = start + length # End *length* bins later
    wvl_regions = []
    flux_regions = []
    nmasks = []
    i = 0
    while i < len(bounds):
        lb, hb = bounds[i]
        ## If window was too short to account for the span, we split window and
        ## start over
        if end+edges > len(wvl):
            ## If out of bounds, change the start of the window 
            i+=1 # We simply pass to the next iteration
        elif (hb > wvl[end]) & (i==(len(bounds)-1)):
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
                ## If the mask englobs the entirety of the window
                if (wvl[start]>=mask[k][0]) & (wvl[end]<=mask[k][1]):
                    _nmask[:] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            start = end - edges # Safeguards lines ingnored because of the edges
            end = start+length
        ## If we reached the final window, we need not extend boundaries
        elif (i==(len(bounds)-1)):
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            i+=1
        ## If the hb is in the window we look for the next window
        elif hb < wvl[end]: # If hb is in the window
            i+=1
        ## If hb out of window, we chunck a window and start a new one
        else:
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            start = np.where(wvl[wvl<lb])[0][-1]
            end = start+length
            i+=1
    return wvl_regions, flux_regions, nmasks


def make_regions_order_revamp(wvl, flux, bounds, mask, length=400):
    '''New function just like make_regions_order, but we try a method more robust.
    In this version, I iteratively merge the regions that are the closest to one another 
    untill I reach regions of a maximum length.
    Known issue 1: the regions may end on the edges of the region. Should now be solved
    Known issue 2: the lines may be added to two regions. Solved
    
    Performance optimizations:
    - Use vectorized operations where possible
    - Pre-compute wavelength condition array to avoid repeated boolean indexing
    - Use searchsorted for efficient bound checking
    - Replace list.remove() with more efficient operations
    '''

    ## Define the mask array that will be returned - vectorized operation
    maskarray = np.zeros(flux.shape, dtype=np.float64)
    if len(mask) > 0:
        mask_arr = np.array(mask)
        for _mask in mask_arr:
            maskarray[(wvl >= _mask[0]) & (wvl <= _mask[1])] = 1

    ## Verify that the bounds are within the order - vectorized clipping
    wvl_min, wvl_max = wvl[0], wvl[-1]
    _used_bounds = []
    for _bounds in bounds:
        lb, hb = _bounds
        # Clip bounds to wavelength range
        lb = max(lb, wvl_min)
        hb = min(hb, wvl_max)
        _used_bounds.append([lb, hb])

    # Optimize merging loop - avoid repeated list.remove() which is O(n)
    merge_threshold = 0.8 * length
    
    while len(_used_bounds) > 1:
        # Compute all spaces at once (vectorized)
        spaces = np.array([_used_bounds[i][1] - _used_bounds[i+1][0] 
                          for i in range(len(_used_bounds)-1)])
        
        # Find candidates for merging
        tried = set()
        merged = False
        
        while len(tried) < len(spaces):
            # Find minimum untried space
            available_indices = [i for i in range(len(spaces)) if i not in tried]
            if not available_indices:
                break
                
            spaces_subset = spaces[available_indices]
            min_idx_in_subset = np.argmin(np.abs(spaces_subset))
            ii = available_indices[min_idx_in_subset]
            
            # Check if merge is valid
            new_bound = [_used_bounds[ii][0], _used_bounds[ii+1][1]]
            
            # Use searchsorted for faster range checking
            start_idx = np.searchsorted(wvl, new_bound[0], side='left')
            end_idx = np.searchsorted(wvl, new_bound[1], side='right')
            len_new_bound = end_idx - start_idx
            
            if len_new_bound < merge_threshold:
                # Perform merge efficiently
                _used_bounds[ii] = new_bound
                _used_bounds.pop(ii+1)
                merged = True
                break
            else:
                tried.add(ii)
        
        if not merged:
            break

    ## Now _used_bounds should contain what we want to put at the center of 400 bins windows
    ## Pre-allocate arrays for better performance
    num_regions = len(_used_bounds)
    wvl_regions = []
    flux_regions = []
    mask_regions = []
    
    wvl_len = len(wvl)
    
    for i in range(num_regions):
        lb, hb = _used_bounds[i]
        
        # Use searchsorted for efficient index finding
        start_idx = np.searchsorted(wvl, lb, side='left')
        end_idx = np.searchsorted(wvl, hb, side='right')
        nbbins = end_idx - start_idx
        
        if nbbins > length:
            raise Exception('Fatal error, we try to create a region that is too large.')
        
        # Center the region
        diff = length - nbbins
        halfdiff = diff // 2
        inival = max(0, start_idx - halfdiff)
        
        # Boundary check
        if inival + length > wvl_len:
            inival = wvl_len - length
        
        # Extract regions using slicing (faster than indexing with arange)
        end_val = inival + length
        _wvl = wvl[inival:end_val].copy()
        _flx = flux[inival:end_val].copy()
        _msk = maskarray[inival:end_val].copy()
        
        # Apply bound mask efficiently
        _msk[(wvl[inival:end_val] < lb) | (wvl[inival:end_val] > hb)] = 0

        wvl_regions.append(_wvl)
        flux_regions.append(_flx)
        mask_regions.append(_msk)
    
    return wvl_regions, flux_regions, mask_regions

def make_regions_order(wvl, flux, bounds, mask, length=400):
    '''
    Make regions from all the information on the lines and regions
    '''
    # edges = int(1/8*length)
    edge = int(1/8*length)
    edge = 0

    ## Is is possible that that one region is longer than required length
    ## We should implement something here at some point.

    ## Define the mask array that will be returned
    maskarray = np.zeros(flux.shape)
    for _mask in mask:
        maskarray[(wvl>_mask[0]) & (wvl<_mask[1])] = 1

    wvl_regions = []; flux_regions = []; mask_regions = [];
    i=0
    while i < len(bounds):
        ## Take the first bound
        lb, hb = bounds[i]
        ## Verify that the bounds are within the order
        if (lb<wvl[0]) | (hb>wvl[-1]):
            i += 1
            continue
        ## We select windows of length bins around the 
        start = np.where(wvl[wvl<lb])[0][-1] # Start as close as possible to bound
        end = start + length
        ## Have we reached the end of the order?
        if end >= len(wvl):
            end = len(wvl) - 1 - edge
            start = end - length
        hbn = hb
        ## Is there a next line? And another one after that?
        ## If yes, does it fall in the same window?
        while (hbn<wvl[end]-edge) & (i+1<(len(bounds))) \
            & (lb>=wvl[0]+edge) & (hbn<wvl[-1]-edge): ## While true, we jump bounds
            if hbn > wvl[end]: ## We don't want to go beyond the region
                break
            else:
                i += 1 ## Next iteration
                lbn, hbn = bounds[i]
            ## at this point, we may have gone too far on the right edge.
        # if hbn > wvl[end]: ## We don't want to go beyond the region
        #     print(hbn, wvl[end])
        #     i -= 1  
        #     lbn, hbn = bounds[i]
        ## Recenter the region
        nbBinsUp = np.sum(wvl[start:end] > hbn)
        nbBinsDown = np.sum(wvl[start:end] < lb)
        nbBinsDiff = nbBinsUp - nbBinsDown
        halfNbBinsDiff = nbBinsDiff // 2
        start -= halfNbBinsDiff
        end -= halfNbBinsDiff
        if start < edge:
            start = edge
            end = edge + length
        ## Append resulting regions
        wvl_regions.append(wvl[start:end])
        flux_regions.append(flux[start:end])
        _maskarray = np.copy(maskarray[start:end])
        _maskarray2 = np.zeros(maskarray[start:end].shape)
        for _mask in mask:
            if (_mask[0]>=lb) & (_mask[1]<=hbn):
                _maskarray2[(wvl[start:end]>=_mask[0]) & (wvl[start:end]<=_mask[1])] = 1
        _maskarray = _maskarray * _maskarray2
        mask_regions.append(_maskarray)
        maskarray[(wvl>=hbn) & (wvl<=hbn)] = 0
        i += 1

    return wvl_regions, flux_regions, mask_regions


def make_regions_2d_orders(wvl, flux, bounds, mask, orders, length=400):
    '''
    Wrapper used to run the make_regions function on a 2D spectrum. The function
    returns regions of given length, containing the lines requested by the 
    bounds. The function attempts to minimize the number windows without 
    changing length, and tries to avoid duplicates. 
    Input parameters:
    - wvl       :   [2D array] Wavelenth solution for the spectrum
    - flux      :   [2D array] Spectrum
    - bounds    :   list or array containing list-pair of bounds used to generate
                    a mask.
    - mask      :   Redundant with bounds. To be deleted in future versions
    - orders    :   List or array indicating in which order to search for the
                    bounds. Must be of same length as bounds.
    - length    :   Length of the output windows.
    
    Performance optimizations:
    - Use list comprehension for flattening instead of nested loops
    - Pre-allocate when possible
    - Vectorize order checking
    '''
    bounds, mask, orders = np.array(bounds), np.array(mask), np.array(orders)

    ## Check that the wvl are increasing - use diff once
    firstwaves = wvl.T[0]
    if np.any(np.diff(firstwaves) < 0):
        raise Exception('make_regions_2d_orders: The orders are not increasing in wavelength.')

    wvl_regions, flux_regions, nmasks = [], [], []
    
    # Process each order
    for order in range(len(flux)):
        # Vectorized order check
        idx = orders == order
        if not np.any(idx):
            continue
        
        _bounds = increase_width(bounds[idx], 0.05)
        _bounds = merge(_bounds)
        _mask = bounds[idx]

        _wvl_regions, _flux_regions, _nmasks = make_regions_order_revamp(wvl[order], 
                                                            flux[order], 
                                                            _bounds, 
                                                            _mask, 
                                                            length=length)

        wvl_regions.append(_wvl_regions)
        flux_regions.append(_flux_regions)
        nmasks.append(_nmasks)

    ## Rebuilt a 1D list of regions - optimized flattening
    # Pre-filter valid regions
    valid_regions = []
    for i in range(len(wvl_regions)):
        for j in range(len(wvl_regions[i])):
            # Skip empty masks
            if np.all(nmasks[i][j] == 0):
                continue
            
            # Check length and collect valid regions
            if len(wvl_regions[i][j]) >= length:
                valid_regions.append((wvl_regions[i][j], flux_regions[i][j], nmasks[i][j]))
            elif len(wvl_regions[i][j]) > 0:
                # Print warning for undersized regions (original behavior)
                print(f"Warning: Region has length {len(wvl_regions[i][j])} < {length}")

    # Convert to arrays efficiently
    if valid_regions:
        nwvl_regions, nflux_regions, nnmasks = zip(*valid_regions)
        return (np.array(nwvl_regions, dtype=float), 
                np.array(nflux_regions, dtype=float), 
                np.array(nnmasks, dtype=float))
    else:
        # Return empty arrays with correct shape
        return (np.empty((0, length), dtype=float),
                np.empty((0, length), dtype=float),
                np.empty((0, length), dtype=float)) 

def make_regions_2d(wvl, flux, bounds, mask, orders, length=400):
    '''
    Wrapper used to run the make_regions function on a 2D spectrum. The function
    returns regions of given length, containing the lines requested by the 
    bounds. The function attempts to minimize the number windows without 
    changing length, and tries to avoid duplicates. 
    Input parameters:
    - wvl       :   [2D array] Wavelenth solution for the spectrum
    - flux      :   [2D array] Spectrum
    - bounds    :   list or array containing list-pair of bounds used to generate
                    a mask.
    - mask      :   Redundant with bounds. To be deleted in future versions
    - orders    :   List or array indicating in which order to search for the
                    bounds. Must be of same length as bounds.
    - length    :   Length of the output windows.
    '''
    bounds, mask, orders = np.array(bounds), np.array(mask), np.array(orders)

    wvl_regions, flux_regions, nmasks = [], [], []

    known_orders = []
    for order in orders:
        if order in known_orders:
            continue
        known_orders.append(order)
        vals = np.where(orders==order)
        _bounds = increase_width(bounds[vals])
        _bounds = merge(_bounds)
        _mask = merge(mask)
        _wvl_regions, _flux_regions, _nmasks = make_regions(wvl[order], 
                                                            flux[order], 
                                                            _bounds, 
                                                            _mask, 
                                                            length=length)
        wvl_regions.append(_wvl_regions)
        flux_regions.append(_flux_regions)
        nmasks.append(_nmasks)

    nwvl_regions = []
    nflux_regions = []
    nnmasks = []
    for i in range(len(wvl_regions)):
        for j in range(len(wvl_regions[i])):
            if np.all(nmasks[i][j]==0):
                continue
            else:
                nwvl_regions.append(wvl_regions[i][j])
                nflux_regions.append(flux_regions[i][j])
                nnmasks.append(nmasks[i][j])

    return np.array(nwvl_regions), np.array(nflux_regions), np.array(nnmasks) 

def read_lines(filename=None, returnall=False):
    if filename is None:
        # filename = paths.irap_tools_data_path+'selected_line_list.txt'
        raise Exception('read_lines: No input file')
    f = open(filename, 'r')
    w, lb, hb, label, ion, orders = [], [], [], [], [], []
    for line in f.readlines():
        if line.strip()[0]=="#": continue
        if line.strip()=="": continue ## Empty line
        # print(line)
        l = line.split()
        w.append(float(l[0]))
        lb.append(float(l[1]))
        hb.append(float(l[2]))
        label.append(l[3])
        ion.append(int(l[4]))
        orders.append(int(l[5]))
    f.close()
    # # Enlarge regions to take some continuum points
    bounds = []
    for i in range(len(hb)):
        _lb = lb[i]
        _hb = hb[i]
        bounds.append([_lb, _hb])

    ## Create the mask actually use for the analysis
    mask = [[lb[i], hb[i]] for i in range(len(lb))]
    if returnall:
        listobj = [np.array(w), np.array(lb), np.array(hb), label, np.array(ion), \
                np.array(orders)]
        outdict = {'wvls':listobj[0], 'lbs':listobj[1], 'hbs':listobj[2],
                   'labels':listobj[3], 'ions':listobj[4], 'orders':listobj[5]}
        return outdict
    else:
        return np.array(bounds), np.array(orders)


def make_contrained_regions(wvl, flux, regions):
    nwvl = []
    nflux = []
    ## let's take additional 10 km/s on each side
    shift = 5/(3*1e5)
    for region in regions:
        idx = ((wvl>(region[0]-region[0]*shift)) 
               & (wvl<(region[1]+region[1]*shift)))
        if len(wvl[idx]) != len(flux[idx]):
            raise Exception('Problem generating regions') 
        nwvl.append(wvl[idx])
        nflux.append(flux[idx])
    return nwvl, nflux

def read_mask(file):
    '''returns the wavelength and associated line name'''
    f = open(file, 'r')
    wvls = []; labels = []; ions = [];
    for line in f.readlines():
        wvls.append(float(line.split()[0]))
        labels.append(line.split()[3])
        ions.append(int(float(line.split()[4])))
    f.close()
    return wvls, labels, ions