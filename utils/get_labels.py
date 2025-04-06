def get_label(in_motorcycle,in_bike,has_helmet):
    if  in_bike and has_helmet:
        return  "Cyclist with helmet"
    
    if  in_bike and not has_helmet:
        return  "Cyclist without helmet"
    
    if  in_motorcycle and has_helmet:
        return  "Motorcyclist with helmet"
    
    if  in_motorcycle  and not has_helmet:
        return  "Motorcyclist without helmet"

    else:
        return "Unknown"