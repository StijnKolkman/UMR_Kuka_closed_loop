from kukavarproxy import KUKA
import time
from .KRL_Pos import KRLPos

if __name__ == "__main__":
    kuka = KUKA("172.31.1.147")
    # test_e6 = krl.KRLPos("FRAME")
    # test_e6 = krl.KRLPos("C3BI_JOG_FRAME")
    #
    # response = kuka.write("$OV_PRO", 8)
    # print("Val:", response)
    #
    #
    # test_e6.set_all(520, 400, 650, 137, 9, -175)
    # x = [50, 50, -50, -50, -50, 0]
    # y = [0, 50, 50, -50, 50, 0]
    # print(kuka.read("$MOVE_STATE"))
    # # print(kuka.write("$ADVANCE", 4))
    # time.sleep(0.1)
    # buffer_max = int(kuka.read("$ACT_ADVANCE"))
    # prev_dist = float(kuka.read("$DIST_NEXT"))
    # kuka.write("$VEL.CP", 0.2)
    # i = 0
    #
    # buffer = 0
    # while i < 30:
    #     dist = float(kuka.read("$DIST_NEXT"))
    #     print("Dist:", dist)
    #     if (dist < prev_dist):
    #         buffer -= 1
    #     if buffer < buffer_max:
    #         test_e6.set_x(500 + x[i%len(x)])
    #         test_e6.set_y(400 + y[i%len(x)])
    #         response = kuka.write(test_e6.get_name(), test_e6.get_KRL_string())
    #         buffer += 1
    #         i += 1
    #         print(kuka.read("$POS_FOR"))
    #         print("Buffer:", buffer)
    #     prev_dist = dist
    #     time.sleep(0.1)



    # while True:
    #     print(kuka.read("$POS_ACT_MES"))
    #     time.sleep(0.1)
    # test_e6.set_x(450)
    # response = kuka.write(test_e6.get_name(), test_e6.get_KRL_string())
    # val = kuka.decode_kuka(response)
    # print("Val:", response)

    act_pos = KRLPos("$POS_ACT_MES")
    act_pos.read_pos(kuka)
    print(act_pos.values)


    # target_pos = KRLPos("C3BI_JOG_FRAME")
    # target_pos.set_x(x)
    #
    # target_pos.read_pos(kuka)
    # target_pos.send(kuka)


    response = kuka.read("$AXIS_ACT")
    print("Val:", response)
    time.sleep(1)
    # print(kuka.read("PING"))
    kuka.disconnect()
    # kuka.close()
    # kuka.disconnect()
    # response = kuka.write(test_e6.get_KRL_string())