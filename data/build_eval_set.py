"""Builds the hand-written eval set (eval_set_handwritten.json).

Questions and reference answers are written by hand from the UG Handbook. Gold chunks are
not hand-picked: each question lists exact phrases from the handbook, and every chunk that
contains one of them is gold. This covers chunk overlap and policies the handbook repeats
in several chapters (e.g. the 75% attendance rule).
"""
import json, os, re, sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "nust_corpus_preprocessed.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "eval_set_handwritten.json")

corpus = json.load(open(INPUT_FILE, encoding='utf-8'))
norm = lambda s: re.sub(r'\s+', ' ', s).lower()
# (topic, question, reference answer, anchor phrases copied from the handbook)
Q = [
 ("Attendance","What is the minimum attendance required to sit the end semester exam of a course?","75%; below that the student cannot sit the exam and gets an XF grade.",["75% attendance is a must","attendance is less than 75%"]),
 ("Attendance","Can a student who got an XF grade appear in the end semester retest?","No. Only an F for academic failure is eligible for retest; XF (low attendance) is not.",["XF Grade awarded based on low attendance is not qualified to appear in the retest"]),
 ("Attendance","How can a student clear an XF grade?","Only by repeating the course whenever it is offered.",["XF grade so obtained shall only be cleared by repetition","XF grade so obtained will only be cleared by repetition"]),
 ("Attendance","What happens if a student is absent for 45 or more consecutive days without informing the university?","Their admission/registration is suspended.",["45 or more consecutive days"]),
 ("Grading","How many grade points is a B+ worth?","3.50",["B+ 3.50"]),
 ("Grading","What minimum CGPA is needed to graduate from BBA?","2.50",["2.50 for BBA"]),
 ("Grading","What are the requirements to be awarded a BE/BS degree?","Complete the prescribed credit hours and achieve a minimum CGPA of 2.00 with at least a C in the final year project.",["has achieved a minimum CGPA of 2.00 with a minimum of C grade in final year"]),
 ("Grading","If I repeat a course, which grade counts towards my CGPA?","Both grades appear on the transcript, but the better grade is used for CGPA.",["better grade will be used in the computation of CGPA"]),
 ("Grading","If I fail a course and then fail it again after repeating, how many F grades are counted?","Only one F.",["number of Fs counted in this case will be one F only"]),
 ("Grading","What is the highest grade I can get in an end semester retest?","D",["The maximum achievable","grade, in this case, will be D"]),
 ("Grading","How is the final grade calculated for a course with both theory and practical credits?","Credit-weighted average of theory and practical percentages, e.g. (70x3 + 90x1)/4 = 75%.",["multiplying the number of credits assigned to each category"]),
 ("Grading","When is a student placed on academic probation?","When CGPA is below 2.00 (not in the first semester) and they don't qualify for withdrawal, or on disciplinary grounds.",["placed on academic probation under the following conditions"]),
 ("Grading","How many F/XF grades lead to withdrawal from NUST?","Seven or more F/XF grades (or five or more in the first semester).",["accumulates seven or more F/XF grades"]),
 ("Grading","How many consecutive probations result in withdrawal?","Four consecutive probations (not applicable to the 7th and 8th semesters).",["four consecutive probations"]),
 ("Grading","Can I get my exam answer book reassessed?","No. Answer books are not reassessed under any circumstances; only rechecking (totals, unmarked portions) is allowed.",["shall not be reassessed under any circumstances"]),
 ("Grading","How much does it cost to get a paper rechecked?","Rs 500 per paper.",["Rs 500/- per paper"]),
 ("Grading","What is the deadline to apply for rechecking a paper?","Within 15 days after the result is declared.",["within 15 days after the declaration of the relevant result"]),
 ("Curriculum","Until which week of a regular semester can I drop a course?","The 8th week (5th week in summer), with HoD recommendation and Dean approval.",["drop a registered course latest by the 8th week"]),
 ("Curriculum","What is the maximum number of W grades I can have in one semester?","Two per semester, and no more than four accumulated at any time.",["maximum of two Ws in a semester"]),
 ("Curriculum","How many credit hours can an undergraduate register for in a regular semester?","Minimum 12, maximum 18 (21 in exceptional circumstances).",["minimum of 12 to a maximum of 18 CHs","minimum number of credit hours is not less than 12"]),
 ("Curriculum","How many courses can I take in a summer semester?","Two (three in exceptional cases, e.g. after an exchange programme).",["maximum of two courses in a Summer Semester","For Summer Semester a maximum of two courses"]),
 ("Curriculum","What is the maximum time allowed to finish a 4-year bachelor's degree?","Seven years.",["minimum and maximum duration of all above","maximum stipulated time of seven years"]),
 ("Curriculum","How long is the mandatory internship and how is it graded?","At least 6 weeks, a 3 credit hour course recorded as Qualified / Not Qualified.",["Mandatory internship (graded course of 3 credit hours)","reflected on the transcript as Qualified / Not Qualified"]),
 ("Curriculum","How long is the internship for Bachelor of Architecture students?","Minimum 14 weeks.",["minimum duration of 14 weeks"]),
 ("Curriculum","How many credit hours is the community service course?","2 credit hours.",["2 -Credit Hour (CHs) Community Service course"]),
 ("Curriculum","If I miss the end semester exam for an acceptable reason, when is the make-up exam held?","Within the first six weeks of the next semester.",["within the first six weeks after the beginning of the subsequent semester"]),
 ("Curriculum","How late can I arrive to an exam and still be allowed in?","No student may join more than thirty minutes after the exam starts.",["No student is allowed to join the examination thirty minutes"]),
 ("Curriculum","How soon after graduating must I apply for a second degree at NUST?","Within one year of graduation.",["within one year of graduation"]),
 ("Fees","How much tuition fee do I pay to keep my registration while deferring a semester?","25% of the tuition fee (full fee if deferring the first semester).",["25% tuition fee to maintain"]),
 ("Fees","What portion of tuition fee is refunded if I withdraw in the second week of classes?","50% (days 8-15); full refund up to day 7, nothing from day 16.",["Half (50%) of Tuition Fee Refund","8th - 15th day of convening"]),
 ("Fees","Is the admission processing fee refundable?","No, under no circumstances.",["Admission P rocessing Fee is not refundable","Admission Processing Fee (Non Refundable)"]),
 ("Fees","What is the per-semester tuition fee for national engineering students?","PKR 171,350.",["171,350"]),
 ("Fees","What is the penalty for paying tuition fee late?","2% of the total payable amount if paid after 15 days of the invoice, up to the end of the 2nd week of semester.",["2% on the total payable amount"]),
 ("Fees","How much is the make-up fee for a missed OHT, mid or end semester exam?","PKR 5,000 per course (USD 40 for international students).",["Make-up Fee for Missed OHT/MSE/ESE"]),
 ("Fees","Can I pay my university fee in installments?","Yes, by applying through the school accounts branch to Fin Dte / fee section.",["Payment in Installments"]),
 ("Fees","Can a scholarship or fee concession be taken away for misconduct?","Yes; full, 75%, 50% or 25% can be forfeited for failure, lack of progress, breach of discipline or misconduct.",["full, 75%, 50%, or 25% of tuition fee"]),
 ("Hostel","What is the monthly rent for a triple-occupancy hostel room?","Rs 6,780 per month.",["Triple occupancy (with community washroom) - Rs 6,780"]),
 ("Hostel","How much is the hostel mess security deposit?","Rs 15,000, refundable.",["Mess Security (Rs 15,000/-)","Mess Security - Rs 15,000"]),
 ("Hostel","How many days of mess leave are allowed per month?","Maximum 12 days.",["maximum of 12 days"]),
 ("Hostel","Are guests allowed to stay in the hostel?","No; guests may be entertained in the cafeterias only.",["Guests (of students and staff) are not allowed","Guests of students or day scholars in the room"]),
 ("Hostel","Can undergraduate hostel residents keep a car?","No.",["UG students are not allowed to keep a car"]),
 ("Hostel","How much notice is needed before vacating the hostel?","At least one week, in writing to the Manager (Hostel).",["minimum a week in advance"]),
 ("Hostel","Is smoking allowed in the hostels?","No, the university is a smoke-free zone.",["Smoking is prohibited in the rooms"]),
 ("Hostel","Which electric appliances can I keep in my hostel room?","Only a bedroom refrigerator and room cooler, with permission and monthly charges.",["No electric appliance will be permitted"]),
 ("Hostel","For how many years can a BS student live in the hostel?","4 years (5 for B.Arch, LLB and MBBS).",["BS and BE Programme for 4 Years"]),
 ("Conduct","What counts as plagiarism under NUST policy?","Representing another's words or ideas as one's own, or failing to attribute quotations, paraphrases or borrowed facts.",["Plagiarism. Intentionally or knowingly representing"]),
 # Not answerable from the corpus: the bot should abstain
 ("Abstention","What is the Wi-Fi password for the boys' hostels?",None,[]),
 ("Abstention","Who won the NUST inter-school cricket tournament in 2023?",None,[]),
 ("Abstention","What is the monthly parking fee for faculty cars?",None,[]),
 ("Abstention","What was the BSCS tuition fee in 2015?",None,[]),
]
out, bad = [], False
for topic, q, ans, anchors in Q:
    gold = [c['metadata']['doc_id'] for c in corpus if any(norm(a) in norm(c['page_content']) for a in anchors)]
    for a in anchors:
        if not any(norm(a) in norm(c['page_content']) for c in corpus):
            print("NO MATCH:", a); bad = True
    print(f"{len(gold):2d} {topic:10s} {q[:60]:60s} {' '.join(g.replace('UG Handbook.pdf_','') for g in gold)}")
    out.append({"question": q, "gold_ids": gold, "answer": ans, "topic": topic, "answerable": ans is not None})
if bad: sys.exit(1)
json.dump(out, open(OUTPUT_FILE, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print(f"Wrote {len(out)} questions to {OUTPUT_FILE}")
