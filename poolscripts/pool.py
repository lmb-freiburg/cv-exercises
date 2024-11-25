"""Parse output from pruntest.sh"""

import argparse


LOG_DIR = "./_logs"  # must match constants.sh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poolfile", type=str,
                        help="output from pruntest.sh",
                        default=LOG_DIR + "/poolout.log")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="more output")
    parser.add_argument("-u", "--user", type=str, help="Ignore this user", default="")
    args = parser.parse_args()

    ignore_users = [args.user]

    # read parsed arguments
    fn = args.poolfile
    verbose = args.verbose

    # open output of pooltest
    fh = open(fn, "rt")

    # iterate output
    users, processes, pools = [], [], []
    model, used, total = 0, 0, 0
    pc = ""

    input_text = fh.read()
    per_pc = input_text.split("-----POOLPC-----")[1:]

    for input_text_per_pc in per_pc:
        input_text_per_pc = input_text_per_pc.replace("\r", "")
        lines = [line for line in input_text_per_pc.splitlines(keepends=False) if line.strip() != ""]
        pc_num = lines[0]
        pc = "tfpool{:02d}".format(int(pc_num))
        if len(lines) == 1:
            print("{:8} {}".format(pc, "offline"))
            continue
        else:
            pc_name_check = lines[1]
            separated_content = input_text_per_pc.split("-----SEPARATOR-----")[1:]
            user_content = separated_content[0]
            users = list(set(line.split(" ")[0] for line in separated_content[0].splitlines(keepends=False)
                             if line.strip() != ""))
            users = [user for user in users if user != args.user]
            user_str = " ".join(users)

            gpu_content = separated_content[1]
            try:
                gpu_load_mb = int(gpu_content.split("MiB")[0].split(" ")[-1])
            except ValueError:
                gpu_load_mb = -1

            free = False
            if users == []:
                free = True

            print("tfpool{:02d} online - users: {}, gpu memory: {}MB".format(int(pc_num), user_str, gpu_load_mb))

            # for content in separated_content:
            #     print(content)

    #
    #
    #     # remove line separators
    #     line = line.replace("\n", "").replace("\r", "")
    #
    #     # skip indicator lines
    #     if line[:5] == "-" * 5:
    #         continue
    #
    #     # read first line min max values
    #     if i == 0:
    #         min_, max_ = (int(a) for a in line.split("limits ")[1].split(" "))
    #         continue
    #
    #     # try to load new model integer
    #     try:
    #         model = int(line)
    #     except ValueError:
    #         mib = line.find("MiB /")
    #         if line.find("tfpool") > -1:
    #             # pool address
    #             pc = line
    #         elif mib > 0:
    #             # gpu
    #             used = int(line[mib - 5:mib])
    #             total = int(line[mib + 6:mib + 11])
    #         else:
    #             # user or process
    #             if line.find("MiB") > -1:
    #                 processes.append(line)
    #             else:
    #                 users.append(line)
    #         continue
    #
    #     # check pc is offline
    #     if pc == "":
    #         print("{:02} {:8} {}".format(model - 1, pc, "offline"))
    #         continue
    #
    #     # delete all processes with user gings
    #     xprocesses = [a for a in processes if
    #                   a[:5] != "gings" and a[:6] != "ml1808"]
    #
    #     # get users
    #     uss = []
    #     for u in users:
    #         while u.find("  ") > - 1:
    #             u = u.replace("  ", " ")
    #         user = u.split(" ")[0]
    #         if user not in ignore_users:
    #             uss.append(user)
    #     uss = list(set(uss))
    #     n_users = len(uss)
    #
    #     # check pool is free
    #     free = False
    #     if n_users == 0 and used < 100:
    #         free = True
    #         pools.append(pc)
    #
    #     # print info line
    #     if verbose:
    #         print("{:02} {:8} {} gpu {:5}/{:5}MB {:3d} processes, "
    #               "users: {}".format(model - 1, pc, "FREE" if free else "BUSY",
    #                                  used, total, len(xprocesses),
    #                                  " ".join(uss)))
    #
    #     # reset running variables
    #     processes, users = [], []
    #     pc = ""
    #     used, total = 0, 0
    #
    # print(" ".join(pools))


if __name__ == '__main__':
    main()
