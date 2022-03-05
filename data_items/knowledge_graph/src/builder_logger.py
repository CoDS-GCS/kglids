class Builder_logger:

    file = open("logs/kgbuilder_log.txt", "a+")
    print('(logs available at logs/kgbuilder_log.txt)\n')

    def get_logger(self):
        return self.file

    def log(self, info):
        self.file.write(info)
        self.file.write("\n")

