def full_labels(scope):
    f = open(config.data_path)
    label = {}
    for line in f.readlines():
        sp = line.split()

        # Document Header
        if line.startswith('doc'):
            docname = sp[0][4:]
            doctype = sp[2]
            # Constrain the data to be in 3 classes
            if doctype in config.typed_tries['scope']:
                label[docname] = doctype
    f.close()
    return label