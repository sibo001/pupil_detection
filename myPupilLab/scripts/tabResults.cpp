#include "ros/ros.h"
#include "std_msgs/String.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <myPupilLab/Result.h>
#include <string.h>
#include <iostream>

using namespace message_filters;
using namespace std;

//Global variables
int emd_true = 0, emd_false=0, fixation_true=0, fixation_false=0, final_true=0, final_false=0, totalEdgeCase =0, totalCorrectEdgeCase =0, totalError = 0, totalError2 = 0;
int correct_Fix =0, correct_Combine =0, wrong_Fix = 0, wrong_Combine = 0;

void callback(const myPupilLab::ResultConstPtr& EMD, const myPupilLab::ResultConstPtr& fixation, const myPupilLab::ResultConstPtr& final){
	std::string currentTruth;
	bool configDone;
	ros::NodeHandle nh;
	if(nh.getParam("/configDone", configDone) && configDone == true){
		if(nh.getParam("/groundTruth", currentTruth)){
			if(EMD->s == currentTruth) emd_true++;
			else emd_false++;

			if(fixation->s == currentTruth) fixation_true++;
			else fixation_false++;

			if(final->s == currentTruth) final_true++;
			else final_false++;

			auto ptr = find(begin(final->class_names),end(final->class_names),currentTruth);
			if(ptr == end(final->class_names)) totalError++;

			auto ptr1 = find(begin(final->class_names),end(final->class_names),"bottle");
			auto ptr2 = find(begin(final->class_names),end(final->class_names),"cup");
			auto ptr3 = find(begin(final->class_names),end(final->class_names),"mouse");
			if(ptr1 == end(final->class_names)||ptr2 == end(final->class_names)||ptr3 == end(final->class_names)){
				totalError2++;
				if(fixation->s == currentTruth) correct_Fix++;
				else wrong_Fix++;

				if(final->s == currentTruth) correct_Combine++;
				else wrong_Combine++;
			}

			if(final->isEdgeCase == true && ptr != end(final->class_names)){
				totalEdgeCase++;
				if(final->s == currentTruth) totalCorrectEdgeCase++;
			}

			cout << "Current Truth: " << currentTruth << endl;
			cout << endl;

			cout << "EMD:" << endl;
			cout << "Positive: " << emd_true << ", Negative: " << emd_false << endl;
			cout << endl;

			cout << "Fixation:" << endl;
			cout << "Positive: " << fixation_true << ", Negative: " << fixation_false << endl;
			cout << endl;

			cout << "Final:" << endl;
			cout << "Positive: " << final_true << ", Negative: " << final_false << endl;
			cout << endl;


			cout << "No. of error in Mask-RCNN detection: " << totalError<< endl;
			cout << "Incomplete detection error: " << totalError2<< endl;
			cout << "Error caused by incomplete detection for fixation: " << wrong_Fix << endl;
			cout << "Error caused by incomplete detection for combine: " << wrong_Combine<< endl;
			cout << endl;

			cout << "Total Edge Case detected: " << totalEdgeCase << endl;
			cout << "Total Correct Edge Case detected: " << totalCorrectEdgeCase << endl;

			cout << "-------------------------------------------------------------------" << endl;
			cout << endl;
		}
		else cout<<"Rosparam /groundTruth is not set!";
	}
}

int main(int argc, char **argv){
	ros::init(argc, argv, "resultsTabulator");

	ros::NodeHandle nh;
	message_filters::Subscriber<myPupilLab::Result> EMD_result_sub(nh, "final_verdict_EMD", 1);
	message_filters::Subscriber<myPupilLab::Result> fixation_result_sub(nh, "final_verdict_fixation", 1);
	message_filters::Subscriber<myPupilLab::Result> true_result_sub(nh, "final_verdict_true", 1);


	typedef sync_policies::ApproximateTime<myPupilLab::Result, myPupilLab::Result, myPupilLab::Result> MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), EMD_result_sub, fixation_result_sub, true_result_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2, _3));

	ros::spin();

	return 0;
}
